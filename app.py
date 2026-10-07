"""Local web app. Run:  python app.py   (opens http://127.0.0.1:7860 in your browser)"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path

import requests
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from core.captions import style_list
from core.downloader import download_full, is_url, video_info
from core.pipeline import rerender_clip
from core.projects import ProjectStore
from core.utils import ROOT, load_config, load_json, merge_config, probe, safe_name

CFG = load_config()
OUTPUTS = Path(CFG["paths"]["outputs"])
UPLOADS = Path(CFG["paths"]["downloads"]) / "uploads"
UPLOADS.mkdir(parents=True, exist_ok=True)
HOST, PORT = "127.0.0.1", 7860

app = FastAPI(title="Viral Clipper")
app.mount("/media", StaticFiles(directory=str(OUTPUTS)), name="media")


def _media_url(path: str) -> str:
    rel = Path(path).resolve().relative_to(OUTPUTS.resolve())
    return "/media/" + rel.as_posix()


def _decorate(summary: dict) -> dict:
    for c in summary.get("clips", []):
        try:
            mtime = int(Path(c["path"]).stat().st_mtime)
            c["url"] = f"{_media_url(c['path'])}?v={mtime}"   # new version after a remake
        except (ValueError, KeyError, OSError):
            c["url"] = ""
    return summary


# ------------------------------------------------------------------ settings
class Settings(BaseModel):
    clips: int = CFG["clips"]["count"]
    min_seconds: float = CFG["clips"]["min_seconds"]
    max_seconds: float = CFG["clips"]["max_seconds"]
    layout: str = CFG["reframe"]["layout"]
    zoom: str = CFG["reframe"].get("zoom", "auto")
    captions: bool = CFG["captions"]["enabled"]
    caption_style: str = CFG["captions"].get("style", "clean")
    caption_language: str = CFG["captions"].get("language", "auto")
    post_language: str = CFG.get("post_caption", {}).get("language", "auto")
    hook: bool = CFG["hook_title"]["enabled"]
    provider: str = CFG["llm"]["provider"]
    model: str = ""
    whisper: str = CFG["transcription"]["model"]
    language: str = CFG["transcription"].get("language") or ""
    trim_start: float = 0.0
    trim_end: float = 0.0


def make_cfg(settings: dict) -> dict:
    s = Settings(**{k: v for k, v in (settings or {}).items() if k in Settings.model_fields})
    llm = {"provider": s.provider}
    if s.model and s.provider in ("ollama", "openai", "anthropic"):
        llm[s.provider] = {"model": s.model}
    return merge_config(CFG, {
        "clips": {"count": s.clips, "min_seconds": s.min_seconds, "max_seconds": s.max_seconds},
        "reframe": {"layout": s.layout, "zoom": s.zoom},
        "llm": llm,
        "transcription": {"model": s.whisper, "language": s.language or None},
        "captions": {"enabled": s.captions, "style": s.caption_style,
                     "language": s.caption_language},
        "post_caption": {"language": s.post_language},
        "hook_title": {"enabled": s.hook},
        "trim": {"start": s.trim_start, "end": s.trim_end},
    })


def _check(s: Settings) -> None:
    if s.min_seconds >= s.max_seconds:
        raise HTTPException(400, "Minimum length must be shorter than maximum length.")
    if not 1 <= s.clips <= 30:
        raise HTTPException(400, "Choose between 1 and 30 clips.")


STORE = ProjectStore(OUTPUTS, make_cfg)


# ------------------------------------------------------------------ projects
class NewProjects(BaseModel):
    sources: list[str]
    settings: Settings = Settings()


class UrlIn(BaseModel):
    url: str


@app.post("/api/video-info")
def api_video_info(body: UrlIn):
    """Preview details and the qualities this video actually offers."""
    if not is_url(body.url):
        raise HTTPException(400, "That is not a link.")
    try:
        return video_info(body.url.strip())
    except Exception as e:
        raise HTTPException(400, f"Could not read this link: {str(e)[:200]}")


class DownloadIn(BaseModel):
    url: str
    height: int = 1080


DOWNLOADS: dict = {"status": "idle", "progress": 0.0, "message": "", "file": ""}


@app.post("/api/download")
def api_download(body: DownloadIn):
    if DOWNLOADS["status"] == "running":
        raise HTTPException(409, "A download is already running.")
    if not is_url(body.url):
        raise HTTPException(400, "That is not a link.")

    def work():
        DOWNLOADS.update(status="running", progress=0.0, message="Starting...", file="")
        try:
            res = download_full(body.url.strip(), int(body.height), CFG,
                                lambda f, m: DOWNLOADS.update(progress=f, message=m))
            DOWNLOADS.update(status="done", progress=1.0, file=res["name"],
                             folder=res["folder"], size=res["size"],
                             message=f"Saved as {res['name']}")
        except Exception as e:
            DOWNLOADS.update(status="error", message=str(e)[:300])

    threading.Thread(target=work, daemon=True).start()
    return {"ok": True}


@app.get("/api/download")
def api_download_status():
    return DOWNLOADS


class AdjustIn(BaseModel):
    x: float = 0.0
    y: float = 0.0
    zoom: float = 1.0


@app.post("/api/projects/{pid}/clips/{index}/adjust")
def api_adjust(pid: str, index: int, body: AdjustIn):
    """Manual framing correction: re-render one clip using its saved plan."""
    p = STORE.get(pid)
    if not p or not p.get("output_dir"):
        raise HTTPException(404, "Project not found.")
    summary_path = Path(p["output_dir"]) / "summary.json"
    if not summary_path.exists():
        raise HTTPException(404, "This project has no clips yet.")
    cfg = make_cfg(p.get("settings") or {})
    try:
        meta = rerender_clip(str(summary_path), index,
                             {"x": max(-1, min(1, body.x)), "y": max(-1, min(1, body.y)),
                              "zoom": max(0.6, min(1.8, body.zoom))}, cfg)
    except IndexError:
        raise HTTPException(404, "Clip not found.")
    except Exception as e:
        raise HTTPException(400, str(e)[:300])
    return _decorate({"clips": [meta]})["clips"][0]


@app.get("/api/projects")
def list_projects():
    return STORE.list()


@app.post("/api/projects")
def add_projects(body: NewProjects):
    sources = [s.strip().strip('"') for s in body.sources if s and s.strip()]
    if not sources:
        raise HTTPException(400, "Paste a YouTube link or add a video file.")
    if len(sources) > 50:
        raise HTTPException(400, "Add at most 50 videos at a time.")
    _check(body.settings)
    return [STORE.add(src, body.settings.model_dump()) for src in dict.fromkeys(sources)]


@app.get("/api/projects/{pid}")
def get_project(pid: str):
    p = STORE.get(pid)
    if not p:
        raise HTTPException(404, "Project not found.")
    p["result"] = None
    summary = Path(p["output_dir"]) / "summary.json" if p.get("output_dir") else None
    if p["status"] == "done" and summary and summary.exists():
        p["result"] = _decorate(load_json(summary))
    return p


@app.get("/api/projects/{pid}/clips")
def project_clips(pid: str):
    """Clips finished so far - works while the project is still running."""
    p = STORE.get(pid)
    if not p:
        raise HTTPException(404, "Project not found.")
    folder = Path(p.get("output_dir") or "")
    summary = folder / "summary.json"
    if summary.exists():
        data = load_json(summary)
        return {"done": True, "title": p["title"], "clips": _decorate(data)["clips"],
                "output_dir": str(folder)}
    clips = []
    if folder.exists():
        for meta_file in sorted(folder.glob("clip_*.meta.json")):
            try:
                clips.append(load_json(meta_file))
            except Exception:
                continue
    return {"done": False, "title": p["title"], "clips": _decorate({"clips": clips})["clips"],
            "output_dir": str(folder)}


@app.post("/api/projects/{pid}/cancel")
def cancel_project(pid: str):
    if not STORE.get(pid):
        raise HTTPException(404, "Project not found.")
    return {"status": STORE.cancel(pid)}


class RetryIn(BaseModel):
    settings: Settings | None = None


@app.post("/api/projects/{pid}/retry")
def retry_project(pid: str, body: RetryIn):
    if not STORE.get(pid):
        raise HTTPException(404, "Project not found.")
    if body.settings:
        _check(body.settings)
    try:
        return STORE.retry(pid, body.settings.model_dump() if body.settings else None)
    except ValueError as e:
        raise HTTPException(409, str(e))


@app.delete("/api/projects/{pid}")
def delete_project(pid: str, files: bool = True):
    try:
        STORE.delete(pid, with_files=files)
    except ValueError as e:
        raise HTTPException(409, str(e))
    return {"ok": True}


# ------------------------------------------------------------------ uploads
@app.post("/api/upload")
async def upload(request: Request, name: str):
    stem, ext = os.path.splitext(name)
    if ext.lower() not in (".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"):
        raise HTTPException(400, "Choose a video file (mp4, mov, mkv, webm, avi).")
    dest = UPLOADS / f"{safe_name(stem, 60)}{ext.lower()}"
    with open(dest, "wb") as f:
        async for chunk in request.stream():
            f.write(chunk)
    if dest.stat().st_size == 0:
        dest.unlink(missing_ok=True)
        raise HTTPException(400, "The file was empty.")
    duration = 0.0
    try:
        duration = probe(str(dest))["duration"]
    except Exception:
        pass
    return {"path": str(dest), "name": name, "size": dest.stat().st_size, "duration": duration}


# ------------------------------------------------------------------ info
@app.get("/api/status")
def status():
    ollama = {"running": False, "models": []}
    try:
        r = requests.get(f"{CFG['llm']['ollama']['url'].rstrip('/')}/api/tags", timeout=2)
        r.raise_for_status()
        ollama = {"running": True, "models": sorted(m["name"] for m in r.json().get("models", []))}
    except Exception:
        pass
    return {
        "ffmpeg": shutil.which("ffmpeg") is not None,
        "ollama": ollama,
        "defaults": {
            "clips": CFG["clips"]["count"], "min_seconds": CFG["clips"]["min_seconds"],
            "max_seconds": CFG["clips"]["max_seconds"], "layout": CFG["reframe"]["layout"],
            "captions": CFG["captions"]["enabled"], "hook": CFG["hook_title"]["enabled"],
            "provider": CFG["llm"]["provider"], "whisper": CFG["transcription"]["model"],
            "language": CFG["transcription"].get("language") or "",
            "zoom": CFG["reframe"].get("zoom", "auto"),
            "caption_style": CFG["captions"].get("style", "clean"),
            "caption_language": CFG["captions"].get("language", "auto"),
            "post_language": CFG.get("post_caption", {}).get("language", "auto"),
            "models": {p: CFG["llm"][p]["model"] for p in ("ollama", "openai", "anthropic")},
        },
        "caption_styles": style_list(),
        "outputs": str(OUTPUTS),
    }


class FolderIn(BaseModel):
    path: str = ""


@app.post("/api/open-folder")
def open_folder(body: FolderIn):
    path = Path(body.path or OUTPUTS).resolve()
    if OUTPUTS.resolve() not in (path, *path.parents):
        raise HTTPException(400, "Only output folders can be opened.")
    if path.is_file():
        path = path.parent
    if sys.platform.startswith("win"):
        os.startfile(str(path))  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])
    return {"ok": True}


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


if __name__ == "__main__":
    url = f"http://{HOST}:{PORT}"
    print(f"\n  Viral Clipper is running at {url}\n  Keep this window open. Press Ctrl+C to stop.\n")
    threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
