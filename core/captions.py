"""Animated captions as .ass subtitles, in several styles and three languages."""
from __future__ import annotations

import re

# Each style: fonts, colours (RGB hex), and how words appear.
# border 1 = outline around letters, 3 = filled box behind the line.
STYLES: dict[str, dict] = {
    "clean": {
        "label": "Clean", "font": "Arial Black", "urdu_font": "Noto Naskh Arabic",
        "size": 76, "color": "FFFFFF", "highlight": "FFE600", "outline_color": "000000",
        "border": 1, "outline": 6, "shadow": 2, "upper": True, "words": 3, "anim": "pop",
    },
    "bold_box": {
        "label": "Bold box", "font": "Arial Black", "urdu_font": "Noto Naskh Arabic",
        "size": 72, "color": "FFFFFF", "highlight": "00E5FF", "outline_color": "000000",
        "border": 3, "outline": 14, "shadow": 0, "upper": True, "words": 3, "anim": "pop",
        "box_alpha": "40",
    },
    "neon": {
        "label": "Neon glow", "font": "Impact", "urdu_font": "Noto Naskh Arabic",
        "size": 80, "color": "FFFFFF", "highlight": "B36BFF", "outline_color": "5B1FB8",
        "border": 1, "outline": 8, "shadow": 4, "upper": True, "words": 2, "anim": "pop",
    },
    "gold": {
        "label": "Gold", "font": "Georgia", "urdu_font": "Noto Nastaliq Urdu",
        "size": 74, "color": "FFD976", "highlight": "FFFFFF", "outline_color": "2A1B00",
        "border": 1, "outline": 6, "shadow": 3, "upper": False, "words": 4, "anim": "rise",
    },
    "minimal": {
        "label": "Minimal", "font": "Verdana", "urdu_font": "Noto Naskh Arabic",
        "size": 62, "color": "FFFFFF", "highlight": "FFFFFF", "outline_color": "000000",
        "border": 1, "outline": 4, "shadow": 1, "upper": False, "words": 5, "anim": "none",
    },
    "karaoke": {
        "label": "Karaoke", "font": "Trebuchet MS", "urdu_font": "Noto Naskh Arabic",
        "size": 78, "color": "FFFFFF", "highlight": "39FF88", "outline_color": "04220F",
        "border": 1, "outline": 7, "shadow": 2, "upper": True, "words": 1, "anim": "pop",
    },
}
DEFAULT_STYLE = "clean"


def style_list() -> list[dict]:
    return [{"id": k, "label": v["label"]} for k, v in STYLES.items()]


def _ass_color(rgb_hex: str, alpha: str = "00") -> str:
    h = rgb_hex.lstrip("#").upper().ljust(6, "0")
    return f"&H{alpha}{h[4:6]}{h[2:4]}{h[0:2]}"


def _ts(t: float) -> str:
    t = max(0.0, t)
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{s:05.2f}"


def _clean(word: str, upper: bool) -> str:
    word = word.replace("{", "(").replace("}", ")").replace("\\", "")
    return word.upper() if upper else word


def _groups(words: list[dict], per_line: int) -> list[list[dict]]:
    groups, cur = [], []
    for i, w in enumerate(words):
        cur.append(w)
        gap = words[i + 1]["start"] - w["end"] if i + 1 < len(words) else 0
        if len(cur) >= per_line or re.search(r"[.!?,;:…۔،؟]$", w["word"]) or gap > 0.6:
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)
    return groups


def resolve_style(cfg: dict, lang: str) -> dict:
    """Merge the chosen preset with the user's own overrides in config.yaml."""
    c = cfg.get("captions", {})
    st = dict(STYLES.get(str(c.get("style", DEFAULT_STYLE)), STYLES[DEFAULT_STYLE]))
    if lang in ("ur", "ar", "fa"):
        st["font"] = c.get("urdu_font") or st["urdu_font"]
        st["upper"] = False
        if "Nastaliq" in st["font"]:
            st["size"] = int(st["size"] * 1.5)      # nastaliq glyphs are small inside their box
            st["words"] = max(3, st["words"])
            st["outline"] = max(4, int(st["outline"] * 0.7))
    for key, cfg_key in (("font", "font"), ("size", "font_size"), ("color", "color"),
                         ("highlight", "highlight_color"), ("outline_color", "outline_color"),
                         ("words", "words_per_line")):
        if c.get(cfg_key) not in (None, "", False):
            st[key] = c[cfg_key]
    if c.get("uppercase") is not None and lang not in ("ur", "ar", "fa"):
        st["upper"] = bool(c["uppercase"])
    return st


def build_ass(words: list[dict], clip_start: float, clip_end: float, cfg: dict,
              title: str | None, layout_middle: bool = False, lang: str = "en",
              word_level: bool = True) -> str:
    """words use source-video timestamps; the output is relative to the clip start."""
    c = cfg.get("captions", {})
    hk = cfg.get("hook_title", {})
    st = resolve_style(cfg, lang)
    W, H = cfg["output"]["width"], cfg["output"]["height"]
    pos = "middle" if layout_middle else c.get("position", "lower")
    align, margin_v = {"lower": (2, int(H * 0.22)), "middle": (5, 0), "upper": (8, int(H * 0.2))}[pos]
    fs = int(st["size"])
    base, hi = _ass_color(st["color"]), _ass_color(st["highlight"])
    outline = _ass_color(st["outline_color"], st.get("box_alpha", "00") if st["border"] == 3 else "00")
    font = st["font"]
    rtl = lang in ("ur", "ar", "fa", "he")
    # letter spacing must be 0 for Urdu/Arabic, otherwise the letters cannot join up
    spacing = 0 if rtl else 1

    lines = [
        "[Script Info]", "ScriptType: v4.00+", f"PlayResX: {W}", f"PlayResY: {H}",
        "WrapStyle: 0", "ScaledBorderAndShadow: yes", "YCbCr Matrix: TV.601", "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, "
        "Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: Cap,{font},{fs},{base},{base},{outline},&H80000000,-1,0,0,0,100,100,{spacing},0,"
        f"{st['border']},{st['outline']},{st['shadow']},{align},70,70,{margin_v},1",
        f"Style: Hook,{font},{int(hk.get('font_size', 64))},{base},{base},"
        f"{_ass_color(st['outline_color'], '20')},&H99000000,-1,0,0,0,100,100,0,0,3,14,0,8,"
        f"80,80,{int(H * 0.09)},1",
        "", "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]

    if hk.get("enabled") and title:
        end = min(float(hk.get("seconds", 3.0)), clip_end - clip_start)
        lines.append(f"Dialogue: 1,{_ts(0)},{_ts(end)},Hook,,0,0,0,,"
                     f"{{\\fad(150,250)}}{_clean(title, False)}")

    if c.get("enabled", True) and words:
        upper = bool(st["upper"])
        for group in _groups(words, int(st["words"])):
            g_end = group[-1]["end"]
            if not word_level or len(group) == 1:
                start, end = group[0]["start"] - clip_start, g_end - clip_start
                text = " ".join(_clean(w["word"], upper) for w in group)
                lines.append(f"Dialogue: 0,{_ts(start)},{_ts(max(start + 0.1, end))},Cap,,0,0,0,,"
                             f"{_anim(st['anim'], hi)}{text}")
                continue
            for k, w in enumerate(group):
                start = w["start"] - clip_start
                end = (group[k + 1]["start"] if k + 1 < len(group) else g_end) - clip_start
                if end <= start:
                    end = start + 0.08
                parts = []
                for j, other in enumerate(group):
                    txt = _clean(other["word"], upper)
                    if j == k:
                        parts.append(f"{{\\c{hi}\\fscx108\\fscy108}}{txt}{{\\c{base}\\fscx100\\fscy100}}")
                    else:
                        parts.append(txt)
                pre = _anim(st["anim"], hi) if k == 0 else ""
                lines.append(f"Dialogue: 0,{_ts(start)},{_ts(end)},Cap,,0,0,0,,{pre}{' '.join(parts)}")
    return "\n".join(lines) + "\n"


def _anim(kind: str, hi: str) -> str:
    if kind == "pop":
        return "{\\t(0,80,\\fscx104\\fscy104)\\t(80,170,\\fscx100\\fscy100)}"
    if kind == "rise":
        return "{\\move(0,18,0,0,0,140)\\fad(120,80)}".replace("\\move(0,18,0,0,0,140)", "") + "{\\fad(140,90)}"
    return ""
