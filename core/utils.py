"""Shared helpers: config, ffmpeg/ffprobe wrappers, filenames, progress."""
from __future__ import annotations

import copy
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable, Optional

import yaml

ROOT = Path(__file__).resolve().parent.parent

ProgressFn = Callable[[float, str], None]


def noop_progress(_fraction: float, message: str) -> None:
    print(f"  {message}")


# ------------------------------------------------------------------ config
def load_config(path: Optional[str | Path] = None) -> dict:
    path = Path(path) if path else ROOT / "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    for key in ("downloads", "outputs", "cache"):
        p = Path(cfg["paths"][key])
        if not p.is_absolute():
            p = ROOT / p
        p.mkdir(parents=True, exist_ok=True)
        cfg["paths"][key] = str(p)
    return cfg


def merge_config(cfg: dict, overrides: dict) -> dict:
    """Deep-merge a dict of overrides into a copy of the config."""
    out = copy.deepcopy(cfg)

    def _merge(dst: dict, src: dict) -> None:
        for k, v in src.items():
            if isinstance(v, dict) and isinstance(dst.get(k), dict):
                _merge(dst[k], v)
            elif v is not None:
                dst[k] = v

    _merge(out, overrides)
    return out


# ------------------------------------------------------------------ ffmpeg
def require_ffmpeg() -> None:
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise RuntimeError(
                f"'{tool}' was not found. Install FFmpeg and make sure it is on your PATH "
                "(Windows: run  winget install Gyan.FFmpeg  then open a new terminal)."
            )


def run(cmd: list[str], cwd: Optional[str] = None) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-15:])
        raise RuntimeError(f"Command failed: {' '.join(cmd[:4])} ...\n{tail}")
    return proc


def probe(path: str) -> dict:
    proc = run(["ffprobe", "-v", "error", "-print_format", "json",
                "-show_streams", "-show_format", path])
    info = json.loads(proc.stdout)
    video = next((s for s in info["streams"] if s["codec_type"] == "video"), None)
    if video is None:
        raise RuntimeError(f"No video stream found in {path}")
    num, den = (video.get("avg_frame_rate") or video.get("r_frame_rate") or "30/1").split("/")
    fps = float(num) / float(den) if float(den) else 30.0
    width, height = int(video["width"]), int(video["height"])
    # respect phone-style rotation metadata
    rotation = 0
    for sd in video.get("side_data_list", []) or []:
        if "rotation" in sd:
            rotation = int(sd["rotation"])
    if abs(rotation) in (90, 270):
        width, height = height, width
    return {
        "width": width,
        "height": height,
        "fps": fps if 1 < fps < 121 else 30.0,
        "duration": float(info["format"].get("duration", 0.0)),
        "has_audio": any(s["codec_type"] == "audio" for s in info["streams"]),
    }


def extract_audio(video: str, wav_out: str) -> str:
    run(["ffmpeg", "-y", "-i", video, "-vn", "-ac", "1", "-ar", "16000",
         "-c:a", "pcm_s16le", wav_out])
    return wav_out


def cut_segment(video: str, start: float, end: float, out: str) -> str:
    """Frame-accurate cut (re-encodes quickly at high quality)."""
    run(["ffmpeg", "-y", "-ss", f"{start:.3f}", "-i", video, "-t", f"{end - start:.3f}",
         "-map", "0:v:0", "-map", "0:a:0?", "-c:v", "libx264", "-preset", "veryfast",
         "-crf", "16", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
         "-avoid_negative_ts", "make_zero", out])
    return out


# ------------------------------------------------------------------ fast frame reading
def iter_frames(path: str, start: float, duration: float, width: int, height: int):
    """Decode a time range straight into BGR frames of the given size with FFmpeg.
    Seeking is frame-accurate and nothing is re-encoded, which is much faster than
    cutting a temporary file first."""
    cmd = ["ffmpeg", "-loglevel", "error", "-ss", f"{start:.3f}", "-i", path,
           "-t", f"{duration:.3f}", "-map", "0:v:0", "-an", "-sn",
           "-vf", f"scale={width}:{height}:flags=area", "-pix_fmt", "bgr24", "-f", "rawvideo", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=width * height * 3 * 4)
    size = width * height * 3
    try:
        import numpy as np
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            yield np.frombuffer(buf, np.uint8).reshape(height, width, 3)
    finally:
        proc.stdout.close()
        proc.kill()
        proc.wait()


# ------------------------------------------------------------------ fonts
FONT_URLS = {
    "Noto Nastaliq Urdu": "https://raw.githubusercontent.com/notofonts/notofonts.github.io/main/"
                          "fonts/NotoNastaliqUrdu/hinted/ttf/NotoNastaliqUrdu-Regular.ttf",
    "Noto Naskh Arabic": "https://raw.githubusercontent.com/notofonts/notofonts.github.io/main/"
                         "fonts/NotoNaskhArabic/hinted/ttf/NotoNaskhArabic-Bold.ttf",
}


def ensure_font(name: str, cache_dir: str) -> Optional[str]:
    """Urdu needs a font that has Urdu letters; download it once if it isn't here."""
    url = FONT_URLS.get(name)
    if not url:
        return None
    folder = Path(cache_dir) / "fonts"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / url.rsplit("/", 1)[-1]
    if path.exists() and path.stat().st_size > 50_000:
        return str(path)
    try:
        import requests
        r = requests.get(url, timeout=90)
        r.raise_for_status()
        if len(r.content) > 50_000:
            path.write_bytes(r.content)
            return str(path)
    except Exception as e:
        print(f"  Could not download the font {name}: {e}")
    return None


def fonts_dir(cache_dir: str) -> Optional[str]:
    folder = Path(cache_dir) / "fonts"
    return str(folder) if folder.exists() and any(folder.glob("*.tt*")) else None


# ------------------------------------------------------------------ video encoder
_ENCODER: Optional[str] = None


def pick_encoder(preference: str = "auto") -> str:
    """Use the graphics card's video encoder when there is one (much faster)."""
    global _ENCODER
    if preference in ("nvenc", "qsv", "amf", "cpu"):
        return preference
    if _ENCODER is None:
        _ENCODER = "cpu"
        for name, codec, pix in (("nvenc", "h264_nvenc", "yuv420p"), ("qsv", "h264_qsv", "nv12"),
                                 ("amf", "h264_amf", "yuv420p")):
            proc = subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i",
                                   "color=black:s=320x320:d=0.2", "-pix_fmt", pix, "-c:v", codec,
                                   "-f", "null", "-"], capture_output=True)
            if proc.returncode == 0:
                _ENCODER = name
                break
    return _ENCODER


def encoder_args(name: str, crf: int, preset: str) -> list[str]:
    q = str(int(crf) + 3)          # hardware encoders need a slightly lower number for equal quality
    if name == "nvenc":
        return ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", q, "-b:v", "0", "-pix_fmt", "yuv420p"]
    if name == "qsv":
        return ["-c:v", "h264_qsv", "-global_quality", q, "-pix_fmt", "nv12"]
    if name == "amf":
        return ["-c:v", "h264_amf", "-quality", "balanced", "-rc", "cqp",
                "-qp_i", q, "-qp_p", q, "-pix_fmt", "yuv420p"]
    return ["-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p"]


# ------------------------------------------------------------------ misc
def safe_name(text: str, max_len: int = 60) -> str:
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE).strip()
    text = re.sub(r"[\s]+", "_", text)
    return (text[:max_len].rstrip("_") or "clip")


def fmt_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"


def env_or(value: str, env_name: str) -> str:
    return value or os.environ.get(env_name, "")


def save_json(obj: Any, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
