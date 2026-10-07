"""Projects: every video you add becomes a project.

Projects are saved in outputs/projects.json so they survive restarts. A single
background worker processes queued projects one at a time (the work is heavy,
running two at once would only make both slower).
"""
from __future__ import annotations

import json
import os
import shutil
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Callable

from .downloader import is_url
from .pipeline import STAGES, run_pipeline
from .utils import load_json

ACTIVE = ("queued", "running")


class Cancelled(BaseException):
    """BaseException so no `except Exception` inside the pipeline swallows it."""


def _stage_of(frac: float) -> str:
    for name, (_a, b) in STAGES.items():
        if frac < b:
            return name
    return "clips"


def _initial_title(source: str) -> str:
    if is_url(source):
        return source.replace("https://", "").replace("http://", "").replace("www.", "")[:80]
    return Path(source).stem


class ProjectStore:
    def __init__(self, outputs: Path, make_cfg: Callable[[dict], dict]):
        self.outputs = Path(outputs)
        self.path = self.outputs / "projects.json"
        self.make_cfg = make_cfg
        self.lock = threading.RLock()
        self.wake = threading.Event()
        self.cancel_ids: set[str] = set()
        self.projects: dict[str, dict] = {}
        self._last_save = 0.0
        self._load()
        self._import_old_runs()
        threading.Thread(target=self._worker, daemon=True, name="project-worker").start()

    # ------------------------------------------------------------ persistence
    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            backup = self.path.with_suffix(".broken.json")
            shutil.copy(self.path, backup)
            print(f"  projects.json could not be read; a copy was saved as {backup.name}")
            return
        for p in data.get("projects", []):
            if p.get("status") == "running":
                p.update(status="interrupted", message="The app was closed while this was running.")
            self.projects[p["id"]] = p

    def _save(self, force: bool = True) -> None:
        now = time.time()
        if not force and now - self._last_save < 3:
            return
        self._last_save = now
        with self.lock:
            data = {"version": 1, "projects": list(self.projects.values())}
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
            os.replace(tmp, self.path)

    def _import_old_runs(self) -> None:
        """Clips made before projects existed show up as finished projects."""
        known = {os.path.normcase(os.path.abspath(p["output_dir"]))
                 for p in self.projects.values() if p.get("output_dir")}
        added = False
        for f in self.outputs.glob("*/summary.json"):
            d = os.path.normcase(os.path.abspath(f.parent))
            if d in known:
                continue
            try:
                s = load_json(f)
            except Exception:
                continue
            t = f.stat().st_mtime
            pid = uuid.uuid4().hex[:8]
            clips = s.get("clips", [])
            self.projects[pid] = {
                "id": pid, "source": s.get("source", ""), "title": s.get("title", f.parent.name),
                "status": "done", "settings": {}, "created": t, "queued_at": t, "started": t,
                "finished": t, "progress": 1.0, "stage": "done",
                "message": f"{len(clips)} clips ready", "log": [], "output_dir": str(f.parent),
                "clips": len(clips), "best_score": max((c.get("score", 0) for c in clips), default=0),
            }
            added = True
        if added:
            self._save()

    # ------------------------------------------------------------ queries
    def _public(self, p: dict, full: bool = False) -> dict:
        out = {k: v for k, v in p.items() if full or k != "log"}
        if p["status"] == "queued":
            queue = sorted((q for q in self.projects.values() if q["status"] == "queued"),
                           key=lambda q: q["queued_at"])
            out["position"] = [q["id"] for q in queue].index(p["id"]) + 1
        end = p.get("finished") or time.time()
        out["elapsed"] = round(end - p["started"], 1) if p.get("started") else 0
        return out

    def list(self) -> list[dict]:
        with self.lock:
            order = {"running": 0, "queued": 1}
            items = sorted(self.projects.values(), key=lambda p: (
                order.get(p["status"], 2),
                p["queued_at"] if p["status"] == "queued" else -(p.get("finished") or p["created"])))
            return [self._public(p) for p in items]

    def get(self, pid: str) -> dict | None:
        with self.lock:
            p = self.projects.get(pid)
            return self._public(p, full=True) if p else None

    # ------------------------------------------------------------ actions
    def add(self, source: str, settings: dict) -> dict:
        now = time.time()
        pid = uuid.uuid4().hex[:8]
        p = {"id": pid, "source": source, "title": _initial_title(source), "status": "queued",
             "settings": settings, "created": now, "queued_at": now, "started": None,
             "finished": None, "progress": 0.0, "stage": "download", "message": "Waiting in queue",
             "log": [], "output_dir": "", "clips": 0, "best_score": 0}
        with self.lock:
            self.projects[pid] = p
            self._save()
        self.wake.set()
        return self._public(p)

    def cancel(self, pid: str) -> str:
        with self.lock:
            p = self.projects[pid]
            if p["status"] == "queued":
                had_clips = p.get("output_dir") and (Path(p["output_dir"]) / "summary.json").exists()
                if had_clips:   # a remake was taken out of the queue: keep the previous clips
                    p.update(status="done", progress=1.0, stage="done", message=f"{p['clips']} clips ready",
                             finished=p.get("prev_finished") or time.time())
                else:
                    p.update(status="cancelled", message="Removed from the queue.", finished=time.time())
                self._save()
            elif p["status"] == "running":
                self.cancel_ids.add(pid)
                p["message"] = "Stopping after the current step..."
            return p["status"]

    def retry(self, pid: str, settings: dict | None = None) -> dict:
        with self.lock:
            p = self.projects[pid]
            if p["status"] in ACTIVE:
                raise ValueError("This project is already queued or running.")
            if settings:
                p["settings"] = settings
            p["prev_finished"] = p.get("finished")
            p.update(status="queued", queued_at=time.time(), started=None, finished=None,
                     progress=0.0, stage="download", message="Waiting in queue", log=[])
            self._save()
        self.wake.set()
        return self._public(p)

    def delete(self, pid: str, with_files: bool = True) -> None:
        with self.lock:
            p = self.projects.get(pid)
            if not p:
                return
            if p["status"] == "running":
                raise ValueError("Stop this project before deleting it.")
            folder = Path(p["output_dir"]) if p.get("output_dir") else None
            del self.projects[pid]
            self._save()
        if with_files and folder and folder.exists():
            root = self.outputs.resolve()
            if root in folder.resolve().parents:          # never delete outside outputs/
                shutil.rmtree(folder, ignore_errors=True)

    # ------------------------------------------------------------ worker
    def _next(self) -> dict | None:
        with self.lock:
            queue = [p for p in self.projects.values() if p["status"] == "queued"]
            return min(queue, key=lambda p: p["queued_at"]) if queue else None

    def _worker(self) -> None:
        while True:
            p = self._next()
            if p is None:
                self.wake.wait(timeout=5)
                self.wake.clear()
                continue
            self._run(p)

    def _run(self, p: dict) -> None:
        pid = p["id"]
        with self.lock:
            p.update(status="running", started=time.time(), progress=0.0, stage="download",
                     message="Starting...", log=[])
            self._save()

        def progress(frac: float, msg: str) -> None:
            if pid in self.cancel_ids:
                raise Cancelled()
            p["progress"] = max(p["progress"], float(frac))
            p["stage"] = _stage_of(p["progress"])
            p["message"] = msg
            if not p["log"] or p["log"][-1] != msg:
                p["log"] = (p["log"] + [msg])[-120:]
            self._save(force=False)

        def on_video(video: dict) -> None:
            p["title"] = video["title"]

        try:
            cfg = self.make_cfg(p["settings"])
            summary = run_pipeline(p["source"], cfg, progress, project_id=pid, on_video=on_video)
            clips = summary["clips"]
            p.update(status="done", progress=1.0, stage="done", output_dir=summary["output_dir"],
                     clips=len(clips), best_score=max((c["score"] for c in clips), default=0),
                     message=f"{len(clips)} clips ready")
        except Cancelled:
            p.update(status="cancelled", message="Stopped.")
        except Exception as e:
            traceback.print_exc()
            p.update(status="error", message=str(e)[:600])
        finally:
            self.cancel_ids.discard(pid)
            p["finished"] = time.time()
            self._save()
