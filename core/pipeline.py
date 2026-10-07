"""Run the whole process: source -> transcript -> viral moments -> reframed vertical clips."""
from __future__ import annotations

import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import language as lang
from .captions import build_ass
from .downloader import get_video
from .reframer import ReframePlan, analyze
from .renderer import render
from .selector import select_clips, write_post_captions
from .transcriber import transcribe
from .utils import (ProgressFn, ensure_font, fmt_time, load_json, noop_progress, probe,
                    require_ffmpeg, safe_name, save_json)

# share of the progress bar each stage gets
STAGES = {"download": (0.00, 0.10), "transcribe": (0.10, 0.38), "select": (0.38, 0.50),
          "clips": (0.50, 1.00)}


def _stage(progress: ProgressFn, name: str) -> ProgressFn:
    a, b = STAGES[name]
    return lambda f, msg: progress(a + (b - a) * max(0.0, min(1.0, f)), msg)


def _caption_words(clip: dict, cfg: dict, spoken: str, translated: dict | None) -> tuple[list, str, bool]:
    """Which words to show as captions, in which language, and whether the
    word-by-word highlight is possible."""
    target = lang.resolve_target(str(cfg["captions"].get("language", "auto")), spoken)
    how = lang.plan_for(target, spoken)
    if how == "transliterate":
        return lang.roman_words(clip["words"]), "roman", True
    if how == "whisper_translate" and translated:
        words = [w for w in (x for s in translated["segments"] for x in s["words"])
                 if clip["start"] - 0.2 <= w["start"] and w["end"] <= clip["end"] + 0.2]
        if words:
            return words, "en", True
    if how == "ai_translate":
        return clip["words"], spoken, True          # not available: keep the spoken language
    return clip["words"], target if how == "as_is" else spoken, True


def run_pipeline(source: str, cfg: dict, progress: ProgressFn = noop_progress,
                 project_id: str | None = None, on_video=None) -> dict:
    """project_id gives the output folder a stable name, so remaking a project
    replaces its old clips. on_video(video_dict) is called once the title is known."""
    require_ffmpeg()
    t0 = time.time()

    video = get_video(source, cfg, _stage(progress, "download"))
    if on_video:
        on_video(video)
    info = probe(video["path"])
    progress(0.10, f"Source: {info['width']}x{info['height']}, {fmt_time(info['duration'])}")

    # the part of the video the user chose with the trimmer
    trim = cfg.get("trim") or {}
    t_start = max(0.0, float(trim.get("start") or 0.0))
    t_end = float(trim.get("end") or 0.0) or info["duration"]
    t_end = min(t_end, info["duration"])
    if t_end - t_start < 5:
        t_start, t_end = 0.0, info["duration"]

    transcript = transcribe(video["path"], video["id"], cfg, _stage(progress, "transcribe"))
    spoken = (transcript.get("language") or "en").lower()

    if t_start > 0 or t_end < info["duration"] - 0.5:
        segs = [s for s in transcript["segments"] if s["end"] > t_start and s["start"] < t_end]
        transcript = {**transcript, "segments": segs}
        progress(0.38, f"Using {fmt_time(t_start)} to {fmt_time(t_end)} of the video")

    clips = select_clips(transcript, cfg, _stage(progress, "select"))
    if not clips:
        raise RuntimeError("No suitable clips were found. Try a longer part of the video, "
                           "or lower the minimum length.")
    write_post_captions(clips, cfg, spoken, lambda f, m: progress(0.48 + 0.02 * f, m))

    # English captions for a non-English video come from Whisper's own translation
    translated = None
    cap_target = lang.resolve_target(str(cfg["captions"].get("language", "auto")), spoken)
    if lang.plan_for(cap_target, spoken) == "whisper_translate":
        progress(0.50, "Translating the speech to English for the captions...")
        try:
            translated = transcribe(video["path"], video["id"], cfg,
                                    lambda f, m: progress(0.50, m), task="translate")
        except Exception as e:
            progress(0.50, f"Translation failed ({e}); captions stay in the spoken language.")
    if cap_target in ("ur", "ar", "fa"):
        ensure_font(cfg["captions"].get("urdu_font") or "Noto Nastaliq Urdu", cfg["paths"]["cache"])
        ensure_font("Noto Naskh Arabic", cfg["paths"]["cache"])

    out_dir = Path(cfg["paths"]["outputs"]) / f"{safe_name(video['title'], 50)}_{project_id or video['id'][-8:]}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in list(out_dir.glob("clip_*.mp4")) + list(out_dir.glob("clip_*.txt")) \
            + list(out_dir.glob("clip_*.plan.json")) + list(out_dir.glob("clip_*.meta.json")) \
            + list(out_dir.glob("summary.json")):
        old.unlink(missing_ok=True)

    tmp = Path(tempfile.mkdtemp(prefix="vclip_"))
    results: list[dict] = []
    pool = ThreadPoolExecutor(max_workers=1)
    prefetched: dict = {}
    clip_stage = _stage(progress, "clips")
    try:
        for n, clip in enumerate(clips, 1):
            base, span = (n - 1) / len(clips), 1 / len(clips)

            def sub(a: float, b: float) -> ProgressFn:
                return lambda f, msg: clip_stage(base + span * (a + (b - a) * f),
                                                 f"Clip {n}/{len(clips)} - {msg}")

            name = f"clip_{n:02d}_{safe_name(clip['title'], 40)}"
            dur = clip["end"] - clip["start"]
            if n - 1 in prefetched:
                sub(0.0, 0.4)(0.5, "analysing faces and motion...")
                plan = prefetched.pop(n - 1).result()
            else:
                plan = analyze(video["path"], cfg, sub(0.0, 0.4), start=clip["start"],
                               duration=dur, info=info)
            if n < len(clips):
                nxt = clips[n]
                prefetched[n] = pool.submit(analyze, video["path"], cfg, lambda f, m: None,
                                            start=nxt["start"], duration=nxt["end"] - nxt["start"],
                                            info=info)

            words, cap_lang, word_level = _caption_words(clip, cfg, spoken, translated)
            split_share = sum(s.f1 - s.f0 for s in plan.scenes if s.layout in ("split", "face_screen")) \
                / max(1, plan.n_frames)
            ass_path = None
            if cfg["captions"].get("enabled") or cfg["hook_title"].get("enabled"):
                ass_path = str(tmp / f"{name}.ass")
                Path(ass_path).write_text(
                    build_ass(words, clip["start"], clip["end"], cfg, clip["title"],
                              layout_middle=split_share > 0.5, lang=cap_lang,
                              word_level=word_level), encoding="utf-8")

            out_mp4 = out_dir / f"{name}.mp4"
            render(video["path"], plan, str(out_mp4), cfg, ass_path, info["has_audio"],
                   sub(0.4, 1.0), start=clip["start"], duration=dur)
            save_json({**plan.to_dict(), "caption_words": words, "caption_lang": cap_lang,
                       "word_level": word_level}, out_dir / f"{name}.plan.json")

            needs_review = plan.coverage < float(cfg["reframe"].get("min_coverage", 0.85))
            meta = {
                "file": out_mp4.name, "name": name, "title": clip["title"], "score": clip["score"],
                "start": clip["start"], "end": clip["end"],
                "duration": round(dur, 1),
                "source_time": f"{fmt_time(clip['start'])} - {fmt_time(clip['end'])}",
                "hook": clip.get("hook", ""), "reason": clip.get("reason", ""),
                "caption": clip.get("caption", ""), "hashtags": clip.get("hashtags", []),
                "caption_language": clip.get("caption_language", ""),
                "subtitle_language": cap_lang,
                "layout": plan.summary(), "zoom": plan.zoom,
                "face_coverage": round(plan.coverage, 3), "needs_review": needs_review,
                "face_detector": plan.detector, "transcript": clip.get("text", ""),
                "adjust": {"x": 0.0, "y": 0.0, "zoom": 1.0},
            }
            meta["path"] = str(out_mp4)
            _write_meta_file(out_dir, name, meta)
            results.append(meta)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
        shutil.rmtree(tmp, ignore_errors=True)

    summary = {"source": source, "title": video["title"], "output_dir": str(out_dir),
               "video_path": video["path"], "spoken_language": spoken,
               "trim": {"start": t_start, "end": t_end},
               "elapsed_seconds": round(time.time() - t0, 1), "clips": results}
    save_json(summary, out_dir / "summary.json")
    progress(1.0, f"Done! {len(results)} clips saved to {out_dir}")
    return summary


def _write_meta_file(out_dir: Path, name: str, meta: dict) -> None:
    save_json({**meta, "path": str(out_dir / meta["file"])}, out_dir / f"{name}.meta.json")
    tags = " ".join(f"#{h}" for h in meta.get("hashtags", []))
    (out_dir / f"{name}.txt").write_text(
        f"TITLE: {meta['title']}\n\nCAPTION:\n{meta.get('caption') or meta['hook']}\n\n{tags}\n\n"
        f"---\nViral score: {meta['score']}/100\nWhy: {meta['reason']}\n"
        f"From source: {meta['source_time']}\nLayout: {meta['layout']} (zoom: {meta.get('zoom', '')})\n"
        f"Face coverage: {meta['face_coverage']:.0%}"
        f"{'  (NEEDS REVIEW)' if meta['needs_review'] else ''}\n\n"
        f"TRANSCRIPT:\n{meta['transcript']}\n", encoding="utf-8")


def rerender_clip(summary_path: str, clip_index: int, adjust: dict, cfg: dict,
                  progress: ProgressFn = noop_progress) -> dict:
    """Re-render one clip with a manual framing correction. No analysis needed:
    the reframing plan was saved when the clip was first made."""
    summary = load_json(summary_path)
    out_dir = Path(summary["output_dir"])
    meta = summary["clips"][clip_index]
    plan_file = out_dir / f"{meta['name']}.plan.json"
    if not plan_file.exists():
        raise RuntimeError("This clip was made by an older version, so it cannot be adjusted. "
                           "Use Remake on the project instead.")
    source = summary.get("video_path")
    if not source or not Path(source).exists():
        raise RuntimeError("The source video is no longer on this PC, so the clip cannot be redone.")

    saved = load_json(plan_file)
    plan = ReframePlan.from_dict(saved)
    info = probe(source)
    tmp = Path(tempfile.mkdtemp(prefix="vclip_adj_"))
    try:
        ass_path = None
        words = saved.get("caption_words") or []
        if words and (cfg["captions"].get("enabled") or cfg["hook_title"].get("enabled")):
            ass_path = str(tmp / f"{meta['name']}.ass")
            Path(ass_path).write_text(
                build_ass(words, meta["start"], meta["end"], cfg, meta["title"],
                          lang=saved.get("caption_lang", "en"),
                          word_level=saved.get("word_level", True)), encoding="utf-8")
        render(source, plan, str(out_dir / meta["file"]), cfg, ass_path, info["has_audio"],
               progress, start=meta["start"], duration=meta["end"] - meta["start"], adjust=adjust)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    meta["adjust"] = adjust
    _write_meta_file(out_dir, meta["name"], meta)
    save_json(summary, summary_path)
    return meta
