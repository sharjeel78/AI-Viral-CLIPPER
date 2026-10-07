"""Caption languages: English, Urdu (Urdu script) and Roman Urdu.

- Roman Urdu is produced by transliterating each Urdu word, so every word keeps its
  exact timing and the word-by-word highlight still works.
- English captions for an Urdu video come from Whisper's own translation, which is
  accurate and free.
- Urdu captions for an English video need the AI, and are timed per caption line.
"""
from __future__ import annotations

import re

URDU_RANGE = re.compile(r"[\u0600-\u06FF\u0750-\u077F]")

# words that a letter-by-letter conversion gets wrong; these are the common ones
WORDS = {
    "یہ": "yeh", "ہے": "hai", "ہیں": "hain", "تھا": "tha", "تھی": "thi", "تھے": "thay",
    "ہو": "ho", "ہوں": "hoon", "ہوتا": "hota", "ہوتی": "hoti", "ہوتے": "hotay",
    "کا": "ka", "کی": "ki", "کے": "ke", "کو": "ko", "کر": "kar", "کرنا": "karna",
    "کرتے": "karte", "کرتی": "karti", "کرتا": "karta", "کریں": "karein", "کیا": "kya",
    "کیوں": "kyun", "کون": "kaun", "کہاں": "kahan", "کب": "kab", "کیسے": "kaise",
    "میں": "mein", "ہم": "hum", "آپ": "aap", "تم": "tum", "وہ": "woh", "اس": "is",
    "ان": "in", "جو": "jo", "اور": "aur", "لیکن": "lekin", "مگر": "magar", "پھر": "phir",
    "نہیں": "nahi", "نہ": "na", "ہاں": "haan", "بھی": "bhi", "سے": "se", "پر": "par",
    "تک": "tak", "ساتھ": "saath", "بغیر": "baghair", "بعد": "baad", "پہلے": "pehle",
    "بہت": "bohat", "زیادہ": "zyada", "کم": "kam", "اچھا": "acha", "اچھی": "achi",
    "برا": "bura", "بڑا": "bara", "چھوٹا": "chota", "نیا": "naya", "پرانا": "purana",
    "ایک": "aik", "دو": "do", "تین": "teen", "چار": "chaar", "پانچ": "paanch",
    "بات": "baat", "باتیں": "baatein", "لوگ": "log", "لوگوں": "logon", "آدمی": "aadmi",
    "کام": "kaam", "وقت": "waqt", "دن": "din", "رات": "raat", "سال": "saal",
    "پیسہ": "paisa", "پیسے": "paise", "زندگی": "zindagi", "دنیا": "duniya",
    "چیز": "cheez", "چیزیں": "cheezein", "طرح": "tarah", "طریقہ": "tareeqa",
    "مطلب": "matlab", "شروع": "shuru", "ختم": "khatam", "سب": "sab", "کچھ": "kuch",
    "سکتا": "sakta", "سکتے": "saktay", "سکتی": "sakti", "چاہیے": "chahiye",
    "گا": "ga", "گی": "gi", "گے": "ge", "رہا": "raha", "رہی": "rahi", "رہے": "rahe",
    "دیکھ": "dekh", "دیکھو": "dekho", "سنو": "suno", "جانتا": "janta", "پتہ": "pata",
    "اصل": "asal", "سچ": "sach", "جھوٹ": "jhoot", "مسئلہ": "masla", "حل": "hal",
    "بھائی": "bhai", "یار": "yaar", "دوست": "dost", "گھر": "ghar", "شہر": "sheher",
    "اللہ": "Allah", "انشاءاللہ": "InshaAllah", "ماشاءاللہ": "MashaAllah",
}

# letter by letter, longest first
LETTERS = [
    ("آ", "aa"), ("اَ", "a"), ("ای", "ai"), ("او", "au"), ("ا", "a"),
    ("بھ", "bh"), ("پھ", "ph"), ("تھ", "th"), ("ٹھ", "th"), ("جھ", "jh"), ("چھ", "ch"),
    ("دھ", "dh"), ("ڈھ", "dh"), ("کھ", "kh"), ("گھ", "gh"), ("ڑھ", "rh"),
    ("ب", "b"), ("پ", "p"), ("ت", "t"), ("ٹ", "t"), ("ث", "s"), ("ج", "j"), ("چ", "ch"),
    ("ح", "h"), ("خ", "kh"), ("د", "d"), ("ڈ", "d"), ("ذ", "z"), ("ر", "r"), ("ڑ", "r"),
    ("ز", "z"), ("ژ", "zh"), ("س", "s"), ("ش", "sh"), ("ص", "s"), ("ض", "z"), ("ط", "t"),
    ("ظ", "z"), ("ع", "a"), ("غ", "gh"), ("ف", "f"), ("ق", "q"), ("ک", "k"), ("گ", "g"),
    ("ل", "l"), ("م", "m"), ("ن", "n"), ("ں", "n"), ("و", "o"), ("ہ", "h"), ("ھ", "h"),
    ("ء", ""), ("ی", "i"), ("ے", "e"), ("ئ", "i"), ("ۂ", "a"), ("ۓ", "e"),
    ("َ", "a"), ("ِ", "i"), ("ُ", "u"), ("ّ", ""), ("ٰ", "a"), ("ْ", ""), ("ٹ", "t"),
]


def is_urdu(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    return sum(bool(URDU_RANGE.match(c)) for c in letters) / len(letters) > 0.4


def to_roman(word: str) -> str:
    """One Urdu word -> Roman Urdu (messaging style)."""
    core = word.strip()
    trail = ""
    while core and core[-1] in "۔،؟!.,?:;\"')":
        trail = {"۔": ".", "،": ",", "؟": "?"}.get(core[-1], core[-1]) + trail
        core = core[:-1]
    if not core:
        return trail
    if core in WORDS:
        return WORDS[core] + trail
    # a word-final ہ sounds like "a" (jumla), not "h"
    if core.endswith("ہ") and len(core) > 2:
        core = core[:-1] + "\u0627"
    out = core
    for src, dst in LETTERS:
        out = out.replace(src, dst)
    out = _add_vowels(out)
    out = re.sub(r"([bcdfghjklmnpqrstvwxyz])\1{2,}", r"\1\1", out, flags=re.I)
    return (out or core) + trail


DIGRAPHS = ("bh", "ph", "th", "jh", "ch", "dh", "kh", "gh", "sh", "zh", "rh")
VOWELS = set("aeiou")


def _units(text: str) -> list[str]:
    out, i = [], 0
    while i < len(text):
        if text[i:i + 2].lower() in DIGRAPHS:
            out.append(text[i:i + 2])
            i += 2
        else:
            out.append(text[i])
            i += 1
    return out


def _add_vowels(text: str) -> str:
    """Urdu script leaves short vowels out. Put an 'a' back where a reader would say one."""
    units = _units(text)
    if len(units) < 2:
        return text
    vowel = lambda u: u[0].lower() in VOWELS
    # a word starting with two consonants usually has a vowel between them (kitab, jumla)
    if not vowel(units[0]) and not vowel(units[1]):
        units = [units[0], "a"] + units[1:]
    # break up long consonant runs (mehnat), putting the vowel before the last one
    out, run = [], []
    for u in units + ["a"]:
        if vowel(u):
            if len(run) >= 3:
                out += run[:-1] + ["a", run[-1]]
            else:
                out += run
            run = []
            out.append(u)
        else:
            run.append(u)
    if out and out[-1] == "a":
        out.pop()
    return "".join(out + run)


def roman_words(words: list[dict]) -> list[dict]:
    """Transliterate a word list, keeping every timing."""
    out = []
    for w in words:
        text = to_roman(w["word"]) if is_urdu(w["word"]) else w["word"]
        if text:
            out.append({**w, "word": text})
    return out


def resolve_target(requested: str, spoken: str) -> str:
    """requested: auto | en | ur | roman -> the language the captions will be in."""
    spoken = (spoken or "en").lower()
    if requested in ("", "auto", None):
        return "ur" if spoken == "ur" else ("en" if spoken == "en" else spoken)
    return requested


def plan_for(target: str, spoken: str) -> str:
    """How to get there: as_is | transliterate | whisper_translate | ai_translate"""
    spoken = (spoken or "en").lower()
    if target == spoken or (target == "roman" and spoken not in ("ur", "hi", "pa", "fa", "ar")):
        return "as_is"
    if target == "roman":
        return "transliterate"
    if target == "en":
        return "whisper_translate"
    return "ai_translate"
