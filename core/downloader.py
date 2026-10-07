"""Get the source video: download a YouTube (or other) URL with yt-dlp, or use a local file."""
from __future__ import annotations

import hashlib
from pathlib import Path

from .utils import ProgressFn, noop_progress


def is_url(text: str) -> bool:
    return text.strip().lower().startswith(("http://", "https://", "www."))


QUALITY_LABELS = [(2160, "4K"), (1440, "2K"), (1080, "1080p"), (720, "720p"), (480, "480p")]


def _ydl(extra: dict | None = None):
    try:
        import yt_dlp
    except ImportError as e:
        raise RuntimeError("yt-dlp is not installed. Run: pip install -U yt-dlp") from e
    opts = {"quiet": True, "no_warnings": True, "noplaylist": True}
    opts.update(extra or {})
    return yt_dlp.YoutubeDL(opts)


def video_info(url: str) -> dict:
    """Title, length, thumbnail and the qualities this video actually has."""
    with _ydl() as ydl:
        info = ydl.extract_info(url, download=False)
    heights = {f.get("height") for f in (info.get("formats") or []) if f.get("height")}
    qualities = [{"height": h, "label": label} for h, label in QUALITY_LABELS
                 if any(x and x >= h - 12 for x in heights)]
    return {
        "id": info.get("id", ""), "title": info.get("title", ""),
        "duration": float(info.get("duration") or 0),
        "thumbnail": info.get("thumbnail", ""), "uploader": info.get("uploader", ""),
        "qualities": qualities or [{"height": 720, "label": "720p"}],
        "url": info.get("webpage_url", url),
    }


def download_full(url: str, height: int, cfg: dict, progress: ProgressFn = noop_progress) -> dict:
    """Download the whole video for the user to keep, at the chosen quality."""
    out_dir = Path(cfg["paths"]["outputs"]) / "downloads"
    out_dir.mkdir(parents=True, exist_ok=True)

    def hook(d: dict) -> None:
        if d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            if total:
                progress(d.get("downloaded_bytes", 0) / total,
                         f"Downloading {height}p... {d.get('downloaded_bytes', 0) / total:.0%}")
        elif d.get("status") == "finished":
            progress(0.98, "Joining video and audio...")

    opts = {
        "format": f"bv*[height<={height}]+ba/b[height<={height}]/b",
        "merge_output_format": "mp4",
        "outtmpl": str(out_dir / f"%(title).80B [%(id)s] {height}p.%(ext)s"),
        "progress_hooks": [hook],
        "concurrent_fragment_downloads": 5,
        "restrictfilenames": True,
    }
    with _ydl(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        path = Path(ydl.prepare_filename(info)).with_suffix(".mp4")
    if not path.exists():
        found = sorted(out_dir.glob(f"*{info['id']}*"), key=lambda p: p.stat().st_mtime)
        if not found:
            raise RuntimeError("The download finished but the file was not found.")
        path = found[-1]
    progress(1.0, "Saved")
    return {"path": str(path), "name": path.name, "size": path.stat().st_size,
            "folder": str(out_dir)}


def get_video(source: str, cfg: dict, progress: ProgressFn = noop_progress) -> dict:
    """Returns {"path", "title", "id"}."""
    source = source.strip().strip('"')
    if not is_url(source):
        p = Path(source)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {source}")
        vid = hashlib.md5(str(p.resolve()).encode()).hexdigest()[:12]
        return {"path": str(p.resolve()), "title": p.stem, "id": f"local_{vid}"}

    try:
        import yt_dlp
    except ImportError as e:
        raise RuntimeError("yt-dlp is not installed. Run: pip install -U yt-dlp") from e

    out_dir = Path(cfg["paths"]["downloads"])
    h = int(cfg["download"].get("max_height", 1080))

    def hook(d: dict) -> None:
        if d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            done = d.get("downloaded_bytes", 0)
            if total:
                progress(done / total, f"Downloading... {done / total:.0%}")

    opts = {
        "format": f"bv*[height<={h}][ext=mp4]+ba[ext=m4a]/bv*[height<={h}]+ba/b[height<={h}]/b",
        "merge_output_format": "mp4",
        "outtmpl": str(out_dir / "%(id)s.%(ext)s"),
        "noplaylist": True,
        "concurrent_fragment_downloads": 5,
        "quiet": True,
        "no_warnings": True,
        "progress_hooks": [hook],
    }
    progress(0.0, "Fetching video info...")
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(source, download=False)
        vid = info["id"]
        existing = list(out_dir.glob(f"{vid}.mp4"))
        if existing:
            progress(1.0, "Video already downloaded - reusing it")
            path = existing[0]
        else:
            ydl.download([source])
            candidates = sorted(out_dir.glob(f"{vid}.*"), key=lambda p: p.stat().st_size, reverse=True)
            candidates = [c for c in candidates if c.suffix in (".mp4", ".mkv", ".webm")]
            if not candidates:
                raise RuntimeError("Download finished but the video file was not found.")
            path = candidates[0]
    return {"path": str(path), "title": info.get("title") or vid, "id": vid}
