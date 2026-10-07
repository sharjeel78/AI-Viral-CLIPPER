"""Turn a ReframePlan into the final 1080x1920 MP4 (with audio and burned-in captions)."""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

import cv2
import numpy as np

from .reframer import ReframePlan
from pathlib import Path

from .utils import (ProgressFn, encoder_args, fonts_dir, iter_frames, noop_progress,
                    pick_encoder)


def _fit_frame(frame: np.ndarray, OW: int, OH: int) -> np.ndarray:
    """Whole frame centred over a blurred, darkened, zoomed copy of itself."""
    h, w = frame.shape[:2]
    s = max(OW / w, OH / h)
    bw, bh = int(w * s / 8) + 1, int(h * s / 8) + 1
    bg = cv2.resize(frame, (bw, bh), interpolation=cv2.INTER_AREA)
    bg = cv2.GaussianBlur(bg, (0, 0), 6)
    bg = cv2.resize(bg, (int(w * s) + 2, int(h * s) + 2), interpolation=cv2.INTER_LINEAR)
    y0, x0 = (bg.shape[0] - OH) // 2, (bg.shape[1] - OW) // 2
    canvas = (bg[y0:y0 + OH, x0:x0 + OW].astype(np.float32) * 0.55).astype(np.uint8)
    fs = min(OW / w, OH / h)
    fw, fh = int(round(w * fs)), int(round(h * fs))
    fg = cv2.resize(frame, (fw, fh), interpolation=cv2.INTER_AREA if fs < 1 else cv2.INTER_LINEAR)
    oy, ox = (OH - fh) // 2, (OW - fw) // 2
    canvas[oy:oy + fh, ox:ox + fw] = fg
    return canvas


def _crop(frame: np.ndarray, cx: float, cy: float, cw: float, ch: float, out_w: int, out_h: int) -> np.ndarray:
    H, W = frame.shape[:2]
    x0 = int(round(min(max(cx - cw / 2, 0), W - cw)))
    y0 = int(round(min(max(cy - ch / 2, 0), H - ch)))
    region = frame[y0:y0 + int(round(ch)), x0:x0 + int(round(cw))]
    interp = cv2.INTER_AREA if region.shape[1] > out_w else cv2.INTER_LINEAR
    return cv2.resize(region, (out_w, out_h), interpolation=interp)


def _native_filters(plan: ReframePlan, OW: int, OH: int, cmd_file: str | None,
                    crop: tuple[int, int]) -> str | None:
    """FFmpeg-only filter graph for clips that are entirely 'track' or entirely 'fit'."""
    layouts = {sc.layout for sc in plan.scenes}
    if layouts == {"track"} and cmd_file:
        cw, ch = crop[0], crop[1]
        return (f"[0:v]sendcmd=f={os.path.basename(cmd_file)},"
                f"crop={cw}:{ch}:0:0,scale={OW}:{OH}:flags=bicubic,setsar=1[v]")
    if layouts == {"fit"}:
        sw, sh = max(2, OW // 8 // 2 * 2), max(2, OH // 8 // 2 * 2)
        return (f"[0:v]split=2[a][b];"
                f"[a]scale={sw}:{sh}:force_original_aspect_ratio=increase,crop={sw}:{sh},boxblur=6:2,"
                f"scale={OW}:{OH},colorchannelmixer=rr=.55:gg=.55:bb=.55[bg];"
                f"[b]scale={OW}:{OH}:force_original_aspect_ratio=decrease[fg];"
                f"[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1[v]")
    return None


def _write_crop_commands(plan: ReframePlan, path: str, cw: int, ch: int,
                         cx_of, cy_of) -> None:
    """Timed commands that move FFmpeg's crop window along the camera path."""
    lines, last = [], None
    max_x, max_y = plan.src_w - cw, plan.src_h - ch
    for f in range(plan.n_frames):
        x = int(round(min(max(cx_of(f) - cw / 2, 0), max_x)))
        y = int(round(min(max(cy_of(f) - ch / 2, 0), max_y)))
        if (x, y) != last:
            lines.append(f"{f / plan.fps:.4f} crop x {x}, crop y {y};")
            last = (x, y)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def render(source_path: str, plan: ReframePlan, out_path: str, cfg: dict,
           ass_path: str | None = None, has_audio: bool = True,
           progress: ProgressFn = noop_progress, start: float = 0.0,
           duration: float | None = None, adjust: dict | None = None) -> str:
    """Render [start, start+duration] of the source using the reframe plan.

    adjust = {"x": -1..1, "y": -1..1, "zoom": 0.6..1.6} shifts or resizes the frame
    manually, as a share of the crop size. It needs no new analysis."""
    a = {"x": 0.0, "y": 0.0, "zoom": 1.0, **(adjust or {})}
    ax, ay, az = float(a["x"]), float(a["y"]), max(0.5, min(2.0, float(a["zoom"])))
    crop_w = max(16.0, min(plan.src_w, plan.crop_w / az))
    crop_h = max(16.0, min(plan.src_h, plan.crop_h / az))
    cx_of = lambda i: float(np.clip(plan.cx[i] + ax * crop_w / 2, crop_w / 2, plan.src_w - crop_w / 2))
    cy_of = lambda i: float(np.clip(plan.cy[i] + ay * crop_h / 2, crop_h / 2, plan.src_h - crop_h / 2))
    ocfg = cfg["output"]
    OW, OH = int(ocfg["width"]), int(ocfg["height"])
    if duration is None:
        duration = plan.n_frames / plan.fps
    crf, preset = int(ocfg.get("crf", 20)), str(ocfg.get("preset", "veryfast"))

    def _encode(encoder: str, with_subs: bool) -> None:
        cmd = ["ffmpeg", "-y", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{OW}x{OH}",
               "-r", f"{plan.fps:.6f}", "-i", "-"]
        if has_audio:
            cmd += ["-ss", f"{start:.3f}", "-t", f"{duration:.3f}", "-i", os.path.abspath(source_path),
                    "-map", "0:v:0", "-map", "1:a:0?"]
        if with_subs and ass_path:
            # run inside the work folder so the filter needs no Windows path escaping
            cmd += ["-vf", f"ass={os.path.basename(ass_path)}:fontsdir=.:shaping=complex"]
        cmd += encoder_args(encoder, crf, preset)
        cmd += ["-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart",
                os.path.abspath(out_path)]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, cwd=work)
        f = 0
        try:
            for frame in iter_frames(source_path, start, duration, plan.src_w, plan.src_h):
                fi = min(f, plan.n_frames - 1)
                si, scene = plan.layout_at(fi)
                if scene.layout == "track":
                    out = _crop(frame, cx_of(fi), cy_of(fi), crop_w, crop_h, OW, OH)
                elif scene.layout == "face_screen":
                    sp = plan.split.get(si, {})
                    fw = min(plan.src_w, float(sp.get("cw", crop_w)) / az)
                    fh = min(plan.src_h, float(sp.get("ch", crop_h / 2)) / az)
                    face = _crop(frame, cx_of(fi), cy_of(fi), fw, fh, OW, OH // 2)
                    screen = _fit_frame(frame, OW, OH - OH // 2)
                    out = np.vstack([face, screen])
                    out[OH // 2 - 3:OH // 2 + 3, :] = 0
                elif scene.layout == "split" and si in plan.split:
                    sp = plan.split[si]
                    k = min(fi - scene.f0, len(sp["ax"]) - 1)
                    top = _crop(frame, sp["ax"][k], sp["ay"][k], sp["cw"], sp["ch"], OW, OH // 2)
                    bot = _crop(frame, sp["bx"][k], sp["by"][k], sp["cw"], sp["ch"], OW, OH - OH // 2)
                    out = np.vstack([top, bot])
                    out[OH // 2 - 3:OH // 2 + 3, :] = 0
                else:
                    out = _fit_frame(frame, OW, OH)
                proc.stdin.write(out.tobytes())
                f += 1
                if f % 15 == 0:
                    progress(min(0.99, f / plan.n_frames), f"Rendering ({encoder})... {min(1, f / plan.n_frames):.0%}")
        except (BrokenPipeError, OSError):
            pass
        finally:
            try:
                proc.stdin.close()
            except OSError:
                pass
        err = proc.stderr.read().decode("utf-8", "replace")
        if proc.wait() != 0:
            raise RuntimeError(err.strip().splitlines()[-1] if err.strip() else "ffmpeg failed")

    def _encode_native(graph: str, encoder: str, with_subs: bool, cwd: str) -> None:
        if with_subs and ass_path:
            graph = graph[:-3] + f",ass={os.path.basename(ass_path)}:fontsdir=.:shaping=complex[v]"
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-nostats", "-progress", "pipe:1",
               "-ss", f"{start:.3f}", "-t", f"{duration:.3f}", "-i", os.path.abspath(source_path),
               "-filter_complex", graph, "-map", "[v]", "-map", "0:a:0?", "-r", f"{plan.fps:.6f}"]
        cmd += encoder_args(encoder, crf, preset)
        cmd += ["-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", os.path.abspath(out_path)]
        proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, encoding="utf-8", errors="replace")
        for line in proc.stdout:                    # live progress from FFmpeg
            if line.startswith("out_time_us=") or line.startswith("out_time_ms="):
                try:
                    done = int(line.split("=")[1]) / 1e6
                    progress(min(0.99, done / duration), f"Rendering ({encoder}, fast)... {min(1, done / duration):.0%}")
                except ValueError:
                    pass
        err = proc.stderr.read()
        if proc.wait() != 0:
            raise RuntimeError(err.strip().splitlines()[-1] if err.strip() else "ffmpeg failed")

    work = tempfile.mkdtemp(prefix="vclip_r_")
    try:
        if ass_path:
            shutil.copy(ass_path, os.path.join(work, os.path.basename(ass_path)))
        src_fonts = fonts_dir(cfg["paths"]["cache"])
        if src_fonts:
            for font in Path(src_fonts).glob("*.tt*"):
                shutil.copy(font, os.path.join(work, font.name))
        cmd_file = os.path.join(work, "crop_cmds.txt")
        cw_i, ch_i = int(crop_w) // 2 * 2, int(crop_h) // 2 * 2
        if {sc.layout for sc in plan.scenes} == {"track"}:
            _write_crop_commands(plan, cmd_file, cw_i, ch_i, cx_of, cy_of)
        graph = _native_filters(plan, OW, OH, cmd_file, (cw_i, ch_i))

        encoder = pick_encoder(str(ocfg.get("encoder", "auto")))
        attempts = []
        if graph:
            attempts += [("native", encoder, True)] + ([("native", "cpu", True)] if encoder != "cpu" else [])
        attempts += [("python", encoder, True)] + ([("python", "cpu", True)] if encoder != "cpu" else [])
        if ass_path:
            attempts.append(("python", "cpu", False))    # captions failed: render without them
        last = None
        for mode, enc, subs in attempts:
            try:
                if mode == "native":
                    _encode_native(graph, enc, subs and bool(ass_path), work)
                else:
                    _encode(enc, subs and bool(ass_path))
                progress(1.0, "Rendered")
                return out_path
            except RuntimeError as e:
                last = e
                progress(0.0, f"Render attempt ({mode}, {enc}{'' if subs else ', no captions'}) failed: {e}")
        raise RuntimeError(f"Rendering failed: {last}")
    finally:
        shutil.rmtree(work, ignore_errors=True)
