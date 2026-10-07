"""Smart 16:9 -> 9:16 reframing.

Pipeline for one clip:
  1. Read every frame: tiny colour histogram for frame-accurate scene-cut detection.
  2. Every Nth frame (analysis_fps): detect faces + measure motion.
  3. Track faces across frames so each person keeps a stable ID.
  4. Measure mouth activity per person to find the active speaker.
  5. Choose a layout per scene: track (moving crop), split (two speakers stacked)
     or fit (whole frame over a blurred background).
  6. Build a smooth "virtual camera" path with a dead zone, speed limit and
     hard cuts on scene changes / speaker switches.
  7. Score how well the speaker stayed in frame (coverage).
The result is a plan that renderer.py turns into the final vertical video.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .utils import ProgressFn, iter_frames, noop_progress, probe

YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
             "face_detection_yunet/face_detection_yunet_2023mar.onnx")

# how much of the frame height the crop keeps, per zoom level
ZOOM_FACTORS = {"wide": 1.0, "medium": 0.78, "close": 0.62}


# ====================================================================== detection
class FaceDetector:
    """YuNet (accurate, fast) with an automatic fallback to OpenCV's Haar cascade."""

    def __init__(self, cache_dir: str):
        self.kind = "haar"
        self.yunet = None
        model = Path(cache_dir) / "face_detection_yunet_2023mar.onnx"
        if not model.exists() or model.stat().st_size < 10_000:
            try:
                import requests
                r = requests.get(YUNET_URL, timeout=60)
                r.raise_for_status()
                if len(r.content) > 10_000:
                    model.write_bytes(r.content)
            except Exception:
                pass
        if model.exists() and model.stat().st_size > 10_000 and hasattr(cv2, "FaceDetectorYN"):
            try:
                self.yunet = cv2.FaceDetectorYN.create(str(model), "", (320, 320), 0.6, 0.3, 5000)
                self.kind = "yunet"
            except Exception:
                self.yunet = None
        if self.yunet is None:
            path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
            self.haar = cv2.CascadeClassifier(path)

    def _detect(self, bgr: np.ndarray, gray: np.ndarray, min_face: int):
        h, w = bgr.shape[:2]
        if w < 24 or h < 24:
            return []
        if self.yunet is not None:
            self.yunet.setInputSize((w, h))
            _, faces = self.yunet.detect(bgr)
            if faces is None:
                return []
            return [(float(f[0]), float(f[1]), float(f[2]), float(f[3])) for f in faces if f[3] >= min_face]
        found = self.haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6,
                                           minSize=(min_face, min_face))
        return [(float(x), float(y), float(fw), float(fh)) for (x, y, fw, fh) in found]

    def detect(self, bgr: np.ndarray, gray: np.ndarray, rois: list | None = None):
        """Full-frame scan, or cheap scans in small regions around the faces found last time."""
        h, w = bgr.shape[:2]
        min_face = max(16, int(h * 0.055))
        if not rois:
            return self._detect(bgr, gray, min_face)
        out = []
        for (rx, ry, rw, rh) in rois:
            x0, y0 = int(max(0, rx)), int(max(0, ry))
            x1, y1 = int(min(w, rx + rw)), int(min(h, ry + rh))
            if x1 - x0 < 24 or y1 - y0 < 24:
                continue
            for (fx, fy, fw, fh) in self._detect(bgr[y0:y1, x0:x1], gray[y0:y1, x0:x1],
                                                 max(12, int(min_face * 0.8))):
                out.append((fx + x0, fy + y0, fw, fh))
        return out


# ====================================================================== tracking
@dataclass
class Track:
    tid: int
    boxes: dict = field(default_factory=dict)      # sample index -> (x, y, w, h) full-res
    activity: dict = field(default_factory=dict)   # sample index -> mouth activity
    last_seen: int = -1
    last_patch: np.ndarray | None = None

    @property
    def hits(self) -> int:
        return len(self.boxes)

    def center(self, i: int) -> tuple[float, float]:
        x, y, w, h = self.boxes[i]
        return x + w / 2, y + h / 2


def _iou(a, b) -> float:
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def _mouth_patch(gray_small: np.ndarray, box_small) -> np.ndarray | None:
    """Lower-middle part of the face box, normalised to a fixed size."""
    x, y, w, h = box_small
    x0, x1 = int(x + w * 0.22), int(x + w * 0.78)
    y0, y1 = int(y + h * 0.62), int(y + h * 0.98)
    H, W = gray_small.shape
    x0, x1, y0, y1 = max(0, x0), min(W, x1), max(0, y0), min(H, y1)
    if x1 - x0 < 6 or y1 - y0 < 4:
        return None
    patch = cv2.resize(gray_small[y0:y1, x0:x1], (32, 18), interpolation=cv2.INTER_AREA)
    return cv2.equalizeHist(patch).astype(np.float32)


# ====================================================================== plan
@dataclass
class Scene:
    f0: int                 # first frame (inclusive)
    f1: int                 # last frame (exclusive)
    layout: str = "track"
    tracks: list = field(default_factory=list)


@dataclass
class ReframePlan:
    src_w: int
    src_h: int
    fps: float
    n_frames: int
    crop_w: int
    crop_h: int
    scenes: list
    cx: np.ndarray                       # per-frame crop centre x
    cy: np.ndarray                       # per-frame crop centre y
    split: dict = field(default_factory=dict)   # scene idx -> dict of per-frame arrays
    coverage: float = 1.0
    detector: str = ""
    speaker_switches: int = 0
    zoom: str = "wide"

    def layout_at(self, f: int) -> tuple[int, Scene]:
        for i, s in enumerate(self.scenes):
            if s.f0 <= f < s.f1:
                return i, s
        return len(self.scenes) - 1, self.scenes[-1]

    def summary(self) -> str:
        counts: dict[str, int] = {}
        for s in self.scenes:
            counts[s.layout] = counts.get(s.layout, 0) + (s.f1 - s.f0)
        total = max(1, sum(counts.values()))
        names = {"track": "follow", "split": "split", "fit": "full frame",
                 "face_screen": "face+screen"}
        parts = [f"{names.get(k, k)} {v / total:.0%}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])]
        return ", ".join(parts)

    # ------------------------------------------------ saving and loading
    def to_dict(self) -> dict:
        return {
            "src_w": self.src_w, "src_h": self.src_h, "fps": self.fps, "n_frames": self.n_frames,
            "crop_w": self.crop_w, "crop_h": self.crop_h, "coverage": self.coverage,
            "detector": self.detector, "speaker_switches": self.speaker_switches, "zoom": self.zoom,
            "cx": [round(float(v), 1) for v in self.cx],
            "cy": [round(float(v), 1) for v in self.cy],
            "scenes": [{"f0": s.f0, "f1": s.f1, "layout": s.layout} for s in self.scenes],
            "split": {str(k): {kk: ([round(float(x), 1) for x in vv]
                                    if isinstance(vv, np.ndarray) else float(vv))
                               for kk, vv in v.items()} for k, v in self.split.items()},
        }

    @staticmethod
    def from_dict(d: dict) -> "ReframePlan":
        split = {int(k): {kk: (np.array(vv, dtype=float) if isinstance(vv, list) else float(vv))
                          for kk, vv in v.items()} for k, v in d.get("split", {}).items()}
        return ReframePlan(
            src_w=d["src_w"], src_h=d["src_h"], fps=d["fps"], n_frames=d["n_frames"],
            crop_w=d["crop_w"], crop_h=d.get("crop_h", d["src_h"]),
            scenes=[Scene(sc["f0"], sc["f1"], sc["layout"]) for sc in d["scenes"]],
            cx=np.array(d["cx"], dtype=float), cy=np.array(d["cy"], dtype=float), split=split,
            coverage=d.get("coverage", 1.0), detector=d.get("detector", ""),
            speaker_switches=d.get("speaker_switches", 0), zoom=d.get("zoom", "wide"))


# ====================================================================== helpers
def _gauss_smooth(values: np.ndarray, sigma: float) -> np.ndarray:
    if sigma < 0.5 or len(values) < 3:
        return values
    radius = int(3 * sigma)
    k = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    k /= k.sum()
    padded = np.pad(values, radius, mode="edge")
    return np.convolve(padded, k, mode="valid")


def _virtual_camera(targets: np.ndarray, dt: float, span: float, dead_zone: float,
                    max_speed: float, sigma: float) -> np.ndarray:
    """Operator-style camera: hold still inside a dead zone, re-centre with a speed limit."""
    if len(targets) == 0:
        return targets
    # a subject that barely moves gets a locked-off shot
    if np.ptp(targets) < dead_zone * span:
        return np.full_like(targets, float(np.median(targets)))
    cam = np.empty_like(targets)
    pos, moving = float(targets[0]), False
    dz, eps, vmax = dead_zone * span, 0.03 * span, max_speed * span * dt
    for i, t in enumerate(targets):
        diff = t - pos
        if abs(diff) > dz:
            moving = True
        if moving:
            pos += float(np.clip(diff, -vmax, vmax))
            if abs(t - pos) < eps:
                moving = False
        cam[i] = pos
    return _gauss_smooth(cam, sigma)


def _frames_from_samples(sample_frames: np.ndarray, values: np.ndarray, f0: int, f1: int) -> np.ndarray:
    frames = np.arange(f0, f1)
    if len(sample_frames) == 0:
        return np.full(len(frames), np.nan)
    return np.interp(frames, sample_frames, values)


# ====================================================================== main
def analyze(video_path: str, cfg: dict, progress: ProgressFn = noop_progress,
            start: float = 0.0, duration: float | None = None, info: dict | None = None) -> ReframePlan:
    """Analyse the time range [start, start+duration] of the video (default: all of it)."""
    rcfg = cfg["reframe"]
    forced = rcfg.get("layout", "auto")
    zoom = str(rcfg.get("zoom", "auto"))
    info = info or probe(video_path)
    fps = info["fps"]
    W, H = info["width"], info["height"]
    if duration is None:
        duration = max(0.1, info["duration"] - start)
    total_est = max(1, int(round(duration * fps)))

    out_ar = cfg["output"]["width"] / cfg["output"]["height"]        # 0.5625
    half_ar = cfg["output"]["width"] / (cfg["output"]["height"] / 2)  # 9:8 for stacked halves
    already_vertical = W / H <= out_ar * 1.08

    step = max(1, int(round(fps / float(rcfg.get("analysis_fps", 8)))))
    aw = min(W, int(rcfg.get("analysis_width", 640))) // 2 * 2
    scale = W / aw
    ah = max(2, int(round(H / scale)) // 2 * 2)
    detector = FaceDetector(cfg["paths"]["cache"])
    full_every = max(1, int(rcfg.get("full_scan_every", 8)))     # regular full scans keep accuracy

    # ---------------------------------------------------------------- pass 1: read
    hists: list[np.ndarray] = []
    sample_frames: list[int] = []
    detections: list[list] = []           # per sample: [(box_full, box_small)]
    motion_x: list[float] = []            # per sample: motion centroid x (full res) or nan
    gray_samples: list[np.ndarray] = []   # small grayscale per sample (for mouth patches)
    prev_motion = None
    prev_boxes: list = []
    since_full = 999
    f = 0
    for small in iter_frames(video_path, start, duration, aw, ah):   # FFmpeg decodes + downsizes
        thumb = cv2.resize(small, (64, 36), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(thumb, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
        hists.append(cv2.normalize(hist, None).flatten())

        if f % step == 0:
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            rois = None
            if prev_boxes and since_full < full_every:
                rois = [(bx - bw * 0.6, by - bh * 0.6, bw * 2.2, bh * 2.2)
                        for (bx, by, bw, bh) in prev_boxes]
            faces = detector.detect(small, gray, rois)
            if rois is None:
                since_full = 0
            elif len(faces) < len(prev_boxes):
                faces = detector.detect(small, gray)      # someone was lost: scan the whole frame
                since_full = 0
            since_full += 1
            prev_boxes = faces
            detections.append([((x * scale, y * scale, w * scale, h * scale), (x, y, w, h))
                               for (x, y, w, h) in faces])
            gray_samples.append(gray)
            # coarse motion map for "no face" shots
            m = cv2.resize(gray, (160, max(1, int(160 * ah / aw)))).astype(np.float32)
            mx = np.nan
            if prev_motion is not None and prev_motion.shape == m.shape:
                diff = np.abs(m - prev_motion)
                diff[diff < 18] = 0
                col = diff.sum(axis=0)
                if col.sum() > diff.size * 1.5:
                    mx = float((col * np.arange(len(col))).sum() / col.sum()) * W / len(col)
            motion_x.append(mx)
            prev_motion = m
            sample_frames.append(f)
        f += 1
        if f % 30 == 0:
            progress(min(0.95, f / total_est) * 0.8, f"Analysing faces and motion... {min(1, f / total_est):.0%}")
    n_frames = f
    if n_frames == 0:
        raise RuntimeError("Could not read any frames from the clip.")
    sample_frames_np = np.array(sample_frames)
    dt = step / fps

    # ---------------------------------------------------------------- crop size (zoom level)
    face_h = [d[0][3] for dets in detections for d in dets]
    med_face = float(np.median(face_h)) if face_h else 0.0
    if zoom in ZOOM_FACTORS:
        chosen = zoom
    else:                                   # auto: tighter framing when the speaker looks small
        ratio = (med_face / H) if med_face else 1.0
        chosen = "close" if ratio < 0.10 else ("medium" if ratio < 0.17 else "wide")
    crop_h = int(round(H * ZOOM_FACTORS[chosen] / 2) * 2)
    crop_w = int(round(min(W, crop_h * out_ar) / 2) * 2)
    if crop_w >= W:                          # source too narrow for this zoom: back off
        crop_w = int(round(W / 2) * 2)
        crop_h = int(round(min(H, crop_w / out_ar) / 2) * 2)

    # ---------------------------------------------------------------- scene cuts
    cuts = [0]
    min_len = int(fps * 0.6)
    for i in range(1, len(hists)):
        corr = cv2.compareHist(hists[i - 1], hists[i], cv2.HISTCMP_CORREL)
        if corr < 0.55 and i - cuts[-1] >= min_len:
            cuts.append(i)
    cuts.append(n_frames)
    scenes = [Scene(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1)]

    progress(0.82, f"Found {len(scenes)} shot(s); tracking people...")
    cx = np.full(n_frames, W / 2.0)
    cy = np.full(n_frames, H / 2.0)
    split_plans: dict[int, dict] = {}
    covered = counted = 0
    switches_total = 0
    hold_samples = max(1, int(float(rcfg.get("speaker_hold_seconds", 1.2)) / dt))
    act_win = max(1, int(0.8 / dt))

    for si, sc in enumerate(scenes):
        idx = [i for i, fr in enumerate(sample_frames) if sc.f0 <= fr < sc.f1]
        if not idx:
            # very short shot between samples: borrow nearest sample
            nearest = int(np.argmin(np.abs(sample_frames_np - sc.f0)))
            idx = [nearest]

        # ------------------------------------------------ tracking inside the shot
        tracks: list[Track] = []
        next_id = 0
        for i in idx:
            dets = detections[i]
            used = set()
            for tr in tracks:
                if i - tr.last_seen > int(1.0 / dt) + 1:
                    continue
                prev = tr.boxes[tr.last_seen]
                best, best_score = None, 0.0
                for d, (full, _sm) in enumerate(dets):
                    if d in used:
                        continue
                    iou = _iou(prev, full)
                    dist = np.hypot((prev[0] + prev[2] / 2) - (full[0] + full[2] / 2),
                                    (prev[1] + prev[3] / 2) - (full[1] + full[3] / 2))
                    s = iou + (0.5 if dist < prev[2] * 0.6 else 0.0)
                    if s > best_score:
                        best, best_score = d, s
                if best is not None and best_score > 0.15:
                    used.add(best)
                    full, sm = dets[best]
                    tr.boxes[i] = full
                    patch = _mouth_patch(gray_samples[i], sm)
                    if patch is not None and tr.last_patch is not None and tr.last_seen == i - 1:
                        tr.activity[i] = float(np.mean(np.abs(patch - tr.last_patch))) / 255.0
                    tr.last_patch = patch
                    tr.last_seen = i
            for d, (full, sm) in enumerate(dets):
                if d not in used:
                    t = Track(next_id)
                    next_id += 1
                    t.boxes[i] = full
                    t.last_patch = _mouth_patch(gray_samples[i], sm)
                    t.last_seen = i
                    tracks.append(t)
        # drop flickering false positives
        min_hits = max(2, int(len(idx) * 0.15))
        tracks = [t for t in tracks if t.hits >= min_hits]
        tracks.sort(key=lambda t: np.mean([b[0] for b in t.boxes.values()]))
        sc.tracks = tracks

        # ------------------------------------------------ active speaker per sample
        def smoothed_activity(tr: Track, i: int) -> float:
            vals = [tr.activity[j] for j in range(i - act_win, i + act_win + 1) if j in tr.activity]
            return float(np.mean(vals)) if vals else 0.0

        speaker: list[Track | None] = []
        cur: Track | None = None
        since_switch = hold_samples
        switches = 0
        for i in idx:
            present = [t for t in tracks if i in t.boxes]
            if not present:
                speaker.append(None)
                since_switch += 1
                continue
            scored = sorted(((smoothed_activity(t, i), t.boxes[i][2], t) for t in present),
                            key=lambda s: (s[0], s[1]), reverse=True)
            best_act, _bw, best = scored[0]
            if cur is None or cur not in present:
                # nobody talking: largest face; otherwise the most active one
                cand = best if best_act > 0.01 else max(present, key=lambda t: t.boxes[i][2])
                if cur is not None and cand is not cur:
                    switches += 1
                cur, since_switch = cand, 0
            elif best is not cur and since_switch >= hold_samples:
                cur_act = smoothed_activity(cur, i)
                if best_act > 0.012 and best_act > cur_act * 1.35:
                    cur, since_switch = best, 0
                    switches += 1
            since_switch += 1
            speaker.append(cur)
        switches_total += switches

        # ------------------------------------------------ layout choice
        dur = max(dt, (sc.f1 - sc.f0) / fps)
        face_frac = sum(1 for i in idx if any(i in t.boxes for t in tracks)) / len(idx)
        big = [t for t in tracks if t.hits >= 0.6 * len(idx)]
        if forced in ("track", "split", "fit", "face_screen"):
            layout = forced
        elif already_vertical or face_frac < 0.3:
            layout = "fit"
        elif len(big) >= 2 and switches / dur >= float(rcfg.get("split_switch_rate", 0.35)):
            a, b = big[0], big[-1]
            common = [i for i in idx if i in a.boxes and i in b.boxes]
            far = common and np.median([abs(a.center(i)[0] - b.center(i)[0]) for i in common]) > crop_w * 0.8
            layout = "split" if far else "track"
        else:
            layout = "track"
        if layout == "split" and len(tracks) < 2:
            layout = "track" if tracks else "fit"
        if layout == "face_screen" and not tracks:
            layout = "fit"
        if already_vertical and layout in ("track", "split"):
            layout = "fit"
        sc.layout = layout

        # ------------------------------------------------ camera paths
        sf = np.array([sample_frames[i] for i in idx], dtype=float)
        sigma = float(rcfg.get("smoothing_seconds", 0.6)) / dt / 2
        dz, vmax = float(rcfg.get("dead_zone", 0.12)), float(rcfg.get("max_pan_speed", 0.35))

        if layout in ("track", "face_screen"):
            # face_screen frames the face into the top half (9:8), track uses the whole crop
            if layout == "track":
                fw, fh = float(crop_w), float(crop_h)
            else:                                   # face_screen: size the window to the face
                heights = [bx[3] for t in tracks for bx in t.boxes.values()] or [H * 0.25]
                fh = float(min(H, max(np.median(heights) * 3.0, H * 0.38)))
                fw = float(min(W, fh * half_ar))
                fh = fw / half_ar
            tx, ty, lastx, lasty = [], [], W / 2.0, H / 2.0
            for k, i in enumerate(idx):
                present = [t for t in tracks if i in t.boxes]
                if present:
                    xs = [t.boxes[i][0] for t in present] + [t.boxes[i][0] + t.boxes[i][2] for t in present]
                    if max(xs) - min(xs) < fw * 0.85:        # everyone fits: frame the group
                        lastx = (min(xs) + max(xs)) / 2
                        lasty = float(np.mean([t.center(i)[1] for t in present]))
                    elif speaker[k] is not None:
                        lastx, lasty = speaker[k].center(i)
                    lasty += fh * 0.10                        # headroom above the head
                elif not np.isnan(motion_x[i]):
                    lastx = 0.7 * lastx + 0.3 * motion_x[i]
                tx.append(lastx)
                ty.append(lasty)
            tx, ty = np.array(tx), np.array(ty)
            # hard cuts where the speaker changes and the jump is large
            bounds = [0]
            for k in range(1, len(idx)):
                if speaker[k] is not speaker[k - 1] and speaker[k] is not None \
                        and abs(tx[k] - tx[k - 1]) > fw * 0.3:
                    bounds.append(k)
            bounds.append(len(idx))
            for b0, b1 in zip(bounds[:-1], bounds[1:]):
                px = _virtual_camera(tx[b0:b1], dt, fw, dz, vmax, sigma)
                py = _virtual_camera(ty[b0:b1], dt, fh, dz, vmax, sigma)
                fa = sc.f0 if b0 == 0 else int(sf[b0])
                fb = sc.f1 if b1 == len(idx) else int(sf[b1])
                cx[fa:fb] = _frames_from_samples(sf[b0:b1], px, fa, fb)
                cy[fa:fb] = _frames_from_samples(sf[b0:b1], py, fa, fb)
            if layout == "face_screen":
                split_plans[si] = {"cw": float(fw), "ch": float(fh)}
            # coverage: was the speaker's face fully inside the frame?
            for k, i in enumerate(idx):
                spk = speaker[k]
                if spk is None or i not in spk.boxes:
                    continue
                x, y, w, h = spk.boxes[i]
                fr = min(n_frames - 1, sample_frames[i])
                counted += 1
                if (x >= cx[fr] - fw / 2 - w * 0.1 and x + w <= cx[fr] + fw / 2 + w * 0.1
                        and y >= cy[fr] - fh / 2 - h * 0.15 and y + h <= cy[fr] + fh / 2 + h * 0.15):
                    covered += 1

        elif layout == "split":
            a, b = tracks[0], tracks[-1]
            face_med = np.median([bx[3] for t in (a, b) for bx in t.boxes.values()])
            ch = float(min(H, max(face_med * 3.4, H * 0.45)))
            cw = min(ch * half_ar, W / 2.0)
            ch = cw / half_ar
            arrays = {}
            for name, tr in (("a", a), ("b", b)):
                xs, ys, lx, ly = [], [], tr.center(next(iter(tr.boxes)))[0], tr.center(next(iter(tr.boxes)))[1]
                for i in idx:
                    if i in tr.boxes:
                        lx, ly = tr.center(i)
                    xs.append(lx)
                    ys.append(ly + ch * 0.06)
                px = _virtual_camera(np.array(xs), dt, cw, dz, vmax, sigma)
                py = _virtual_camera(np.array(ys), dt, ch, dz, vmax, sigma)
                arrays[name + "x"] = np.clip(_frames_from_samples(sf, px, sc.f0, sc.f1), cw / 2, W - cw / 2)
                arrays[name + "y"] = np.clip(_frames_from_samples(sf, py, sc.f0, sc.f1), ch / 2, H - ch / 2)
            arrays["cw"], arrays["ch"] = cw, ch
            split_plans[si] = arrays
            counted += len(idx)
            covered += len(idx)
        else:  # fit - whole frame is visible
            counted += len(idx)
            covered += len(idx)

    cx = np.clip(cx, crop_w / 2, max(crop_w / 2, W - crop_w / 2))
    cy = np.clip(cy, crop_h / 2, max(crop_h / 2, H - crop_h / 2))
    progress(1.0, "Reframing plan ready")
    return ReframePlan(src_w=W, src_h=H, fps=fps, n_frames=n_frames, crop_w=crop_w, crop_h=crop_h,
                       scenes=scenes, cx=cx, cy=cy, split=split_plans,
                       coverage=(covered / counted) if counted else 1.0, zoom=chosen,
                       detector=detector.kind, speaker_switches=switches_total)
