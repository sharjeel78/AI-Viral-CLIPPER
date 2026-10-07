"""Find the moments with the most viral potential.

The transcript is split into chunks and each chunk is sent to an LLM that returns
candidate clips as JSON. Candidates are then validated, snapped to sentence
boundaries, de-duplicated and ranked. If no LLM is available, a simple heuristic
is used so the tool still produces clips.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

import requests

from .transcriber import all_words
from .utils import ProgressFn, env_or, noop_progress

SYSTEM_PROMPT = """You are an expert short-form video editor who has produced thousands of viral TikToks, Reels and YouTube Shorts.
You find moments inside long videos that will perform well as standalone vertical clips."""

USER_PROMPT = """Below is part of a video transcript. Each line starts with its timestamp in seconds.

Find the {n} best moments to turn into standalone short-form clips of {min_s}-{max_s} seconds.

A great clip:
- has a HOOK in the first 3 seconds (a bold claim, question, surprising fact, strong emotion, conflict)
- makes sense on its own, without the rest of the video
- has a payoff: punchline, insight, story resolution, or strong opinion
- starts at the beginning of a sentence and ends at the end of one
- is not an intro, outro, sponsor read or "like and subscribe"

Score each clip 0-100 for viral potential. Be strict: most moments are 40-70, only exceptional ones are 85+.

Respond with ONLY a JSON object, no other text, in exactly this format:
{{"clips": [{{"start": 123.4, "end": 178.9, "score": 82, "title": "short catchy title (max 8 words)", "hook": "why the first seconds grab attention", "reason": "why it will perform well", "hashtags": ["tag1", "tag2", "tag3"]}}]}}

TRANSCRIPT:
{transcript}"""


# ------------------------------------------------------------------ LLM calls
def _json_complete(text: str) -> bool:
    """True once the model has written a full JSON object (so we can stop it early)."""
    t = text.strip()
    if not t.endswith("}"):
        return False
    try:
        json.loads(t[t.find("{"):])
        return True
    except (json.JSONDecodeError, ValueError):
        return False


def _ollama(system: str, prompt: str, cfg: dict, report) -> str:
    c = cfg["llm"]["ollama"]
    max_out = int(cfg["llm"].get("max_output_tokens", 1200))
    # size the context to the prompt: less memory and faster than always using the maximum
    est = int(len(system + prompt) / 2.5) + max_out + 400
    ctx = min(int(c.get("context", 8192)), max(2048, -(-est // 1024) * 1024))
    body = {
        "model": c["model"], "stream": True, "format": "json", "keep_alive": "30m",
        "options": {"num_ctx": ctx, "temperature": 0.4, "num_predict": max_out},
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
    }
    text, n, t_first = "", 0, None
    with requests.post(f"{c['url'].rstrip('/')}/api/chat", json=body, stream=True,
                       timeout=(10, 900)) as r:
        if r.status_code == 404:
            raise RuntimeError(f"Ollama model '{c['model']}' is not downloaded. Run: ollama pull {c['model']}")
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            msg = json.loads(line)
            if msg.get("error"):
                raise RuntimeError(msg["error"])
            piece = (msg.get("message") or {}).get("content", "")
            if piece:
                if t_first is None:
                    t_first = time.time()
                text += piece
                n += 1
                if n % 10 == 0:
                    rate = n / max(0.1, time.time() - t_first)
                    report(f"writing its picks... {n} tokens ({rate:.1f}/s)")
                if "}" in piece and _json_complete(text):
                    break                         # answer finished: don't wait for trailing spaces
            if msg.get("done"):
                break
    return text


def _call_llm(system: str, prompt: str, cfg: dict, report=lambda m: None) -> str:
    lcfg = cfg["llm"]
    provider = lcfg.get("provider", "ollama")

    if provider == "ollama":
        return _ollama(system, prompt, cfg, report)

    if provider == "openai":
        c = lcfg["openai"]
        key = env_or(c.get("api_key", ""), "LLM_API_KEY")
        r = requests.post(f"{c['base_url'].rstrip('/')}/chat/completions", timeout=300,
                          headers={"Authorization": f"Bearer {key}"}, json={
            "model": c["model"], "temperature": 0.4,
            "max_tokens": int(lcfg.get("max_output_tokens", 1200)) + 800,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": prompt}],
        })
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    if provider == "anthropic":
        c = lcfg["anthropic"]
        key = env_or(c.get("api_key", ""), "ANTHROPIC_API_KEY")
        r = requests.post("https://api.anthropic.com/v1/messages", timeout=300, headers={
            "x-api-key": key, "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }, json={
            "model": c["model"], "max_tokens": 4000, "system": system,
            "messages": [{"role": "user", "content": prompt}],
        })
        r.raise_for_status()
        return "".join(b.get("text", "") for b in r.json()["content"])

    raise ValueError(f"Unknown LLM provider: {provider}")


def _parse_json(text: str) -> list[dict]:
    text = re.sub(r"```(?:json)?", "", text).strip()
    start, end = text.find("{"), text.rfind("}")
    candidates = [text[start:end + 1]] if start != -1 and end > start else []
    lb, rb = text.find("["), text.rfind("]")
    if lb != -1 and rb > lb:
        candidates.append(text[lb:rb + 1])
    for c in candidates:
        try:
            data = json.loads(c)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            if "start" in data and "end" in data:
                data = [data]
            else:
                data = data.get("clips") or data.get("moments") or data.get("results") or []
        if isinstance(data, list):
            found = [d for d in data if isinstance(d, dict) and "start" in d and "end" in d]
            if found:
                return found
    return []


# ------------------------------------------------------------------ chunks
def _chunks(transcript: dict, minutes: float) -> list[str]:
    size = minutes * 60
    overlap = 60.0
    segs = transcript["segments"]
    if not segs:
        return []
    total = segs[-1]["end"]
    out, t = [], 0.0
    while t < total:
        lines = [f"[{s['start']:.1f}] {s['text']}" for s in segs
                 if s["end"] > t and s["start"] < t + size]
        if lines:
            out.append("\n".join(lines))
        t += size - overlap
    return out


# ------------------------------------------------------------------ snapping
_SENT_END = re.compile(r"[.!?…]['\")\]]*$")


def _snap(clip: dict, words: list[dict], min_s: float, max_s: float) -> dict | None:
    if not words:
        return None
    start, end = float(clip["start"]), float(clip["end"])
    if end <= start:
        return None

    # start: first word at/after the requested start, then back up to a sentence start
    si = next((i for i, w in enumerate(words) if w["start"] >= start - 0.4), None)
    if si is None:
        return None
    for back in range(1, 12):
        j = si - back
        if j < 0 or words[si]["start"] - words[j]["start"] > 4:
            break
        if _SENT_END.search(words[j]["word"]):
            si = j + 1
            break

    # end: last word finishing before the requested end, then extend to sentence end
    ei = max((i for i, w in enumerate(words) if w["end"] <= end + 0.4), default=None)
    if ei is None or ei <= si:
        return None
    if not _SENT_END.search(words[ei]["word"]):
        for k in range(ei + 1, min(ei + 25, len(words))):
            if words[k]["end"] - words[si]["start"] > max_s:
                break
            if _SENT_END.search(words[k]["word"]):
                ei = k
                break

    # enforce length limits
    while ei > si and words[ei]["end"] - words[si]["start"] > max_s:
        ei -= 1
        # prefer cutting at a sentence end when trimming
        for k in range(ei, si, -1):
            if words[k]["end"] - words[si]["start"] <= max_s and _SENT_END.search(words[k]["word"]):
                ei = k
                break
    dur = words[ei]["end"] - words[si]["start"]
    if dur < min_s * 0.8:
        return None

    out = dict(clip)
    out["start"] = round(max(0.0, words[si]["start"] - 0.15), 2)
    out["end"] = round(words[ei]["end"] + 0.35, 2)
    return out


def _dedupe(clips: list[dict]) -> list[dict]:
    kept: list[dict] = []
    for c in sorted(clips, key=lambda c: c["score"], reverse=True):
        clash = False
        for k in kept:
            inter = min(c["end"], k["end"]) - max(c["start"], k["start"])
            if inter > 0.3 * min(c["end"] - c["start"], k["end"] - k["start"]):
                clash = True
                break
        if not clash:
            kept.append(c)
    return kept


# ------------------------------------------------------------------ heuristic
def _heuristic(transcript: dict, n: int, min_s: float, max_s: float) -> list[dict]:
    """No-AI fallback: favour dense, emotional, question-heavy stretches."""
    segs = transcript["segments"]
    hot = re.compile(r"\b(never|always|secret|crazy|insane|mistake|truth|actually|nobody|"
                     r"everyone|biggest|worst|best|money|why|how|love|hate|shocking)\b", re.I)
    target = (min_s + max_s) / 2
    cands = []
    for i, s in enumerate(segs):
        text, j = "", i
        while j < len(segs) and segs[j]["end"] - s["start"] < target:
            text += " " + segs[j]["text"]
            j += 1
        if j >= len(segs):
            break
        dur = segs[j]["end"] - s["start"]
        wps = len(text.split()) / max(dur, 1)
        score = (min(wps, 4) * 12 + len(hot.findall(text)) * 6
                 + text.count("?") * 5 + text.count("!") * 5)
        first = segs[i]["text"]
        if "?" in first or hot.search(first):
            score += 10
        cands.append({"start": s["start"], "end": segs[j]["end"], "score": min(95, int(score)),
                      "title": " ".join(first.split()[:8]), "hook": first,
                      "reason": "Selected by built-in heuristic (no AI).",
                      "hashtags": ["shorts", "viral"]})
    return cands


# ------------------------------------------------------------------ post captions
CAPTION_SYSTEM = """You write social media captions for short vertical videos (TikTok, Reels, YouTube Shorts).
You know what makes people stop scrolling, and you write for search as well as for people."""

CAPTION_PROMPT = """Write a social media caption for this short clip.

LANGUAGE: {lang_rule}

Rules:
- First line is a hook: a bold claim, a question or a surprising fact. Keep it under 12 words.
- Then 1 or 2 short lines that make someone want to watch.
- Use words people actually search for on this topic (SEO), naturally, not stuffed.
- End with exactly 5 hashtags: mix broad and specific ones, no spaces inside a hashtag.
- No emojis in the hashtags. At most 3 emojis in the caption text.
- Never invent facts that are not in the clip.

CLIP TITLE: {title}
CLIP TRANSCRIPT: {text}

Answer with ONLY this JSON:
{{"caption": "the caption text with line breaks", "hashtags": ["tag1", "tag2", "tag3", "tag4", "tag5"]}}"""

LANG_RULES = {
    "en": "Write the caption in English.",
    "roman": "Write the caption in Roman Urdu (Urdu written with English letters, the way people "
             "chat, e.g. 'yeh baat sab jantay hain'). Hashtags may be English.",
    "ur": "Write the caption in Urdu script. Hashtags may be English.",
}


def write_post_captions(clips: list[dict], cfg: dict, spoken: str,
                        progress: ProgressFn = noop_progress) -> None:
    """Fill clip["caption"] and clip["hashtags"] with an AI-written post caption."""
    provider = cfg["llm"].get("provider", "ollama")
    want = str(cfg.get("post_caption", {}).get("language", "auto"))
    if want in ("", "auto"):
        want = "roman" if (spoken or "en").lower() in ("ur", "hi", "pa") else "en"
    rule = LANG_RULES.get(want, LANG_RULES["en"])
    cache_dir = Path(cfg["paths"]["cache"]) / "ai_captions"
    cache_dir.mkdir(parents=True, exist_ok=True)

    for i, clip in enumerate(clips):
        clip.setdefault("caption", "")
        text = (clip.get("text") or "")[:2200]
        if provider == "none" or not text:
            clip["caption"] = clip.get("hook") or clip["title"]
            continue
        progress(i / max(1, len(clips)), f"Writing caption for clip {i + 1}/{len(clips)}...")
        prompt = CAPTION_PROMPT.format(lang_rule=rule, title=clip["title"], text=text)
        model = cfg["llm"].get(provider, {}).get("model", "")
        key = hashlib.md5(f"cap|{provider}|{model}|{want}|{prompt}".encode()).hexdigest()
        cached = cache_dir / f"{key}.json"
        try:
            reply = cached.read_text(encoding="utf-8") if cached.exists() else \
                _call_llm(CAPTION_SYSTEM, prompt, cfg)
            data = json.loads(reply[reply.find("{"):reply.rfind("}") + 1])
            caption = str(data.get("caption", "")).strip()
            tags = [re.sub(r"[^\w]", "", str(t)) for t in (data.get("hashtags") or [])]
            tags = [t for t in tags if t][:5]
            if caption:
                clip["caption"] = caption
                if tags:
                    clip["hashtags"] = tags
                clip["caption_language"] = want
                cached.write_text(reply, encoding="utf-8")
        except Exception as e:
            progress(i / max(1, len(clips)), f"Caption AI failed ({e}); using the clip hook.")
            clip["caption"] = clip.get("hook") or clip["title"]


# ------------------------------------------------------------------ main
def select_clips(transcript: dict, cfg: dict, progress: ProgressFn = noop_progress) -> list[dict]:
    ccfg = cfg["clips"]
    n, min_s, max_s = int(ccfg["count"]), float(ccfg["min_seconds"]), float(ccfg["max_seconds"])
    words = all_words(transcript)
    if not words:
        raise RuntimeError("The transcript is empty - no speech was detected in this video.")

    provider = cfg["llm"].get("provider", "ollama")
    cache_dir = Path(cfg["paths"]["cache"]) / "ai_picks"
    cache_dir.mkdir(parents=True, exist_ok=True)
    raw: list[dict] = []
    if provider != "none":
        chunks = _chunks(transcript, float(cfg["llm"].get("chunk_minutes", 10)))
        per_chunk = max(2, min(6, n))
        for i, chunk in enumerate(chunks):
            label = f"part {i + 1}/{len(chunks)}"
            progress(i / len(chunks), f"AI is reading the transcript... {label}")
            prompt = USER_PROMPT.format(n=per_chunk, min_s=int(min_s), max_s=int(max_s),
                                        transcript=chunk)
            model = cfg["llm"].get(provider, {}).get("model", "")
            key = hashlib.md5(f"{provider}|{model}|{SYSTEM_PROMPT}|{prompt}".encode()).hexdigest()
            cache_file = cache_dir / f"{key}.json"
            try:
                if cache_file.exists():          # a remake reuses the AI's earlier answer
                    progress((i + 1) / len(chunks), f"Reusing saved AI picks... {label}")
                    reply = cache_file.read_text(encoding="utf-8")
                else:
                    reply = _call_llm(SYSTEM_PROMPT, prompt, cfg,
                                      lambda m, i=i, label=label: progress(i / len(chunks), f"AI is {m} {label}"))
                    if _parse_json(reply):
                        cache_file.write_text(reply, encoding="utf-8")
                found = _parse_json(reply)
            except Exception as e:
                progress(i / len(chunks), f"AI request failed ({e}). Falling back to heuristic.")
                raw = []
                provider = "none"
                break
            for c in found:
                try:
                    raw.append({
                        "start": float(c["start"]), "end": float(c["end"]),
                        "score": int(float(c.get("score", 50))),
                        "title": str(c.get("title", "")).strip() or "Untitled clip",
                        "hook": str(c.get("hook", "")), "reason": str(c.get("reason", "")),
                        "hashtags": [str(h).lstrip("#") for h in (c.get("hashtags") or [])][:8],
                    })
                except (KeyError, TypeError, ValueError):
                    continue
    if provider == "none" or not raw:
        progress(0.9, "Scoring moments with the built-in heuristic...")
        raw = _heuristic(transcript, n, min_s, max_s)

    snapped = [s for s in (_snap(c, words, min_s, max_s) for c in raw) if s]
    ranked = [c for c in _dedupe(snapped) if c["score"] >= int(ccfg.get("min_score", 0))]
    final = sorted(ranked[:n], key=lambda c: c["start"])
    # attach the words that fall inside each clip (used for captions)
    for c in final:
        c["words"] = [w for w in words if w["start"] >= c["start"] - 0.05 and w["end"] <= c["end"] + 0.05]
        c["text"] = " ".join(w["word"] for w in c["words"])
    progress(1.0, f"Selected {len(final)} clips")
    return final
