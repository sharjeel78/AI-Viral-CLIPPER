"""Speech-to-text with word-level timestamps using faster-whisper (free, runs locally)."""
from __future__ import annotations

import os
from pathlib import Path

from .utils import ProgressFn, extract_audio, load_json, noop_progress, save_json


def _merge_word_pieces(raw_words) -> list[dict]:
    """Whisper can split one written word into several pieces, which is very common in
    Urdu, Hindi and Arabic. A piece that does not start with a space belongs to the word
    before it, so we join them back into whole words (keeping the timings)."""
    out: list[dict] = []
    for w in raw_words:
        text = w.word
        if not text.strip():
            continue
        starts_word = text[:1].isspace() or not out
        gap = (w.start - out[-1]["end"]) if out else 0.0
        if starts_word or gap > 0.35:
            out.append({"start": round(w.start, 3), "end": round(w.end, 3), "word": text.strip()})
        else:
            out[-1]["word"] += text.strip()
            out[-1]["end"] = round(w.end, 3)
    return out


def _load_model(model_size: str, device: str, cache_dir: str):
    from faster_whisper import WhisperModel

    root = os.path.join(cache_dir, "whisper")
    if device in ("auto", "cuda"):
        try:
            return WhisperModel(model_size, device="cuda", compute_type="float16",
                                download_root=root), "cuda"
        except Exception as e:  # no GPU / missing CUDA libraries
            if device == "cuda":
                raise
            print(f"  GPU not available for Whisper ({type(e).__name__}); using CPU.")
    return WhisperModel(model_size, device="cpu", compute_type="int8", download_root=root), "cpu"


def transcribe(video_path: str, video_id: str, cfg: dict,
               progress: ProgressFn = noop_progress, task: str = "transcribe") -> dict:
    """Returns {"language", "duration", "segments": [{start,end,text,words:[{start,end,word}]}]}"""
    tcfg = cfg["transcription"]
    cache_dir = cfg["paths"]["cache"]
    tag = "" if task == "transcribe" else f"_{task}"
    cache_file = Path(cache_dir) / f"transcript_{video_id}_{tcfg['model']}{tag}_v2.json"
    if cache_file.exists():
        progress(1.0, "Transcript found in cache")
        return load_json(cache_file)

    wav = str(Path(cache_dir) / f"audio_{video_id}.wav")
    progress(0.0, "Extracting audio...")
    extract_audio(video_path, wav)

    progress(0.02, f"Loading Whisper '{tcfg['model']}' model (first run downloads it)...")
    model, device = _load_model(tcfg["model"], tcfg.get("device", "auto"), cache_dir)

    def _run(m):
        segs, info = m.transcribe(wav, language=tcfg.get("language") or None, task=task,
                                  word_timestamps=True, vad_filter=True, beam_size=5)
        out = []
        for s in segs:
            words = _merge_word_pieces(s.words or [])
            out.append({"start": round(s.start, 3), "end": round(s.end, 3),
                        "text": s.text.strip(), "words": words})
            if info.duration:
                progress(min(0.99, s.end / info.duration),
                         f"Transcribing ({device})... {s.end / info.duration:.0%}")
        return out, info

    try:
        segments, info = _run(model)
    except Exception as e:
        if device != "cuda":
            raise
        print(f"  GPU transcription failed ({e}); retrying on CPU.")
        model, device = _load_model(tcfg["model"], "cpu", cache_dir)
        segments, info = _run(model)

    result = {"language": info.language, "duration": info.duration, "segments": segments}
    save_json(result, cache_file)
    try:
        os.remove(wav)
    except OSError:
        pass
    progress(1.0, f"Transcribed {len(segments)} segments (language: {info.language})")
    return result


def all_words(transcript: dict) -> list[dict]:
    return [w for s in transcript["segments"] for w in s["words"]]
