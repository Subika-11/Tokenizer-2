# sandhi.py
import regex as re
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Rule:
    pattern: re.Pattern
    repl: str

BOUND = "⟂"  # boundary sentinel

# ================================
# Tamil Sandhi rules (yours, preserved)
# ================================

TA_RULES = [

# ---------------------------------------------------------------------
# A) உயிர் + உயிர் (Vowel–Vowel joins) — coalescence & cue marking
# We mark the boundary before the second vowel (or its onset), so merges don’t cross.
# ---------------------------------------------------------------------

# அ + அ/ஆ … (common a+a ā-type joins)
Rule(re.compile(r"(அ)\s*(அ|ஆ)"), r"\1" + BOUND + r"\2"),
# அ + இ/ஈ  → often /e/-like outcome; mark join
Rule(re.compile(r"(அ)\s*(இ|ஈ)"), r"\1" + BOUND + r"\2"),
# அ + உ/ஊ → often /o/-like; mark join
Rule(re.compile(r"(அ)\s*(உ|ஊ)"), r"\1" + BOUND + r"\2"),
# அ + எ/ஏ, ஒ/ஓ, ஐ/ஔ
Rule(re.compile(r"(அ)\s*(எ|ஏ)"), r"\1" + BOUND + r"\2"),
Rule(re.compile(r"(அ)\s*(ஒ|ஓ)"), r"\1" + BOUND + r"\2"),
Rule(re.compile(r"(அ)\s*(ஐ|ஔ)"), r"\1" + BOUND + r"\2"),

# இ/ஈ + உயிர் (potential y-glide contexts)
Rule(re.compile(r"(இ|ஈ)\s*(அ|ஆ|இ|ஈ|உ|ஊ|எ|ஏ|ஒ|ஓ|ஐ|ஔ)"), r"\1" + BOUND + r"\2"),

# உ/ஊ + உயிர் (potential v-glide contexts)
Rule(re.compile(r"(உ|ஊ)\s*(அ|ஆ|இ|ஈ|உ|ஊ|எ|ஏ|ஒ|ஓ|ஐ|ஔ)"), r"\1" + BOUND + r"\2"),

# எ/ஏ, ஒ/ஓ + உயிர் (diphthong-like joins; keep safe)
Rule(re.compile(r"(எ|ஏ|ஒ|ஓ)\s*(அ|ஆ|இ|ஈ|உ|ஊ|எ|ஏ|ஒ|ஓ|ஐ|ஔ)"), r"\1" + BOUND + r"\2"),

# ஐ/ஔ + உயிர் (mark joins after diphthongs)
Rule(re.compile(r"(ஐ|ஔ)\s*(அ|ஆ|இ|ஈ|உ|ஊ|எ|ஏ|ஒ|ஓ|ஐ|ஔ)"), r"\1" + BOUND + r"\2"),

# ---------------------------------------------------------------------
# B) Glide insertion cues (இடைஎழுத்து தோன்றுதல்) — y/வ positions
# We *mark* the place where a glide typically appears; we don’t insert it.
# Add both independent-vowel and dependent-sign contexts.
# ---------------------------------------------------------------------

# Dependent sign i/ī + அ… (ி/ீ before அ… → y-glide in speech)
Rule(re.compile(r"(ி|ீ)\s*(அ)"), r"\1" + BOUND + r"\2"),
# Dependent sign u/ū + அ… (ு/ூ before அ… → v-glide)
Rule(re.compile(r"(ு|ூ)\s*(அ)"), r"\1" + BOUND + r"\2"),

# Word ends with இ/ஈ, next starts with அ… (independent vowels)
Rule(re.compile(r"(இ|ஈ)\s*(அ)"), r"\1" + BOUND + r"\2"),
# Word ends with உ/ஊ, next starts with அ…
Rule(re.compile(r"(உ|ஊ)\s*(அ)"), r"\1" + BOUND + r"\2"),

# Cases with y/v already present — keep a boundary before the glide
Rule(re.compile(r"(ி|ீ)\s*(ய)"), r"\1" + BOUND + r"\2"),
Rule(re.compile(r"(ு|ூ)\s*(வ)"), r"\1" + BOUND + r"\2"),

# ---------------------------------------------------------------------
# C) Nasal + stop assimilations (மெய் சந்தி)
# We DO NOT rewrite to ங்க/ஞ்ச/ண்ட/ந்த/ம்ப; we just mark the join.
# ---------------------------------------------------------------------

# ங் before க/க-series
Rule(re.compile(r"(ங்)\s*(க)"), r"\1" + BOUND + r"\2"),
# ஞ் before ச/ச-series
Rule(re.compile(r"(ஞ்)\s*(ச)"), r"\1" + BOUND + r"\2"),
# ண் before ட/ட-series
Rule(re.compile(r"(ண்)\s*(ட)"), r"\1" + BOUND + r"\2"),
# ந் before த/த-series
Rule(re.compile(r"(ந்)\s*(த)"), r"\1" + BOUND + r"\2"),
# ம் before ப/ப-series
Rule(re.compile(r"(ம்)\s*(ப)"), r"\1" + BOUND + r"\2"),
# ன் before ந
Rule(re.compile(r"(ன்)\s*(ந)"), r"\1" + BOUND + r"\2"),

# Generic nasal + stop cluster (safety net)
Rule(re.compile(r"(ங்|ஞ்|ண்|ந்|ம்|ன்)\s*(க|ச|ட|த|ப|ற)"), r"\1" + BOUND + r"\2"),

# ---------------------------------------------------------------------
# D) Gemination / doubling across boundary (compounds)
# ---------------------------------------------------------------------

Rule(re.compile(r"(க்)\s*(க)"), r"\1" + BOUND + r"\2"),
Rule(re.compile(r"(ச்)\s*(ச)"), r"\1" + BOUND + r"\2"),
Rule(re.compile(r"(ட்)\s*(ட)"), r"\1" + BOUND + r"\2"),
Rule(re.compile(r"(த்)\s*(த)"), r"\1" + BOUND + r"\2"),
Rule(re.compile(r"(ப்)\s*(ப)"), r"\1" + BOUND + r"\2"),

# Liquids/approximants doubling across boundary
Rule(re.compile(r"(ய்)\s*(ய)"), r"\1" + BOUND + r"\2"),
Rule(re.compile(r"(வ்)\s*(வ)"), r"\1" + BOUND + r"\2"),
Rule(re.compile(r"(ல்)\s*(ல)"), r"\1" + BOUND + r"\2"),
Rule(re.compile(r"(ள்)\s*(ள)"), r"\1" + BOUND + r"\2"),
Rule(re.compile(r"(ர்)\s*(ர)"), r"\1" + BOUND + r"\2"),
Rule(re.compile(r"(ற்)\s*(ற)"), r"\1" + BOUND + r"\2"),
Rule(re.compile(r"(ன்)\s*(ன)"), r"\1" + BOUND + r"\2"),

# ---------------------------------------------------------------------
# E) திரிதல் (mutation) cues — mark classic change environments
# ---------------------------------------------------------------------

# ல் + ச
Rule(re.compile(r"(ல்)\s*(ச)"), r"\1" + BOUND + r"\2"),
# ர்/ற் + ர
Rule(re.compile(r"(ர்)\s*(ர)"), r"\1" + BOUND + r"\2"),
Rule(re.compile(r"(ற்)\s*(ர)"), r"\1" + BOUND + r"\2"),
# Dental↔retroflex interplay triggers
Rule(re.compile(r"(ன்|ண்)\s*(ட|த)"), r"\1" + BOUND + r"\2"),

# ---------------------------------------------------------------------
# F) கெடுதல் (final consonant loss before vowel) — mark likely joins
# ---------------------------------------------------------------------

Rule(re.compile(r"(க்)\s*([அஆஇஈஉஊஎஏஒஓஐஔ])"), r"\1" + BOUND + r"\2"),
Rule(re.compile(r"(ச்)\s*([அஆஇஈஉஊஎஏஒஓஐஔ])"), r"\1" + BOUND + r"\2"),
Rule(re.compile(r"(ட்)\s*([அஆஇஈஉஊஎஏஒஓஐஔ])"), r"\1" + BOUND + r"\2"),
Rule(re.compile(r"(த்)\s*([அஆஇஈஉஊஎஏஒஓஐஔ])"), r"\1" + BOUND + r"\2"),
Rule(re.compile(r"(ப்)\s*([அஆஇஈஉஊஎஏஒஓஐஔ])"), r"\1" + BOUND + r"\2"),

# Final sonorants often reduce/elide before suffix vowels
Rule(re.compile(r"(ம்)\s*([அஆஇஈஉஊஎஏஒஓஐஔ])"), r"\1" + BOUND + r"\2"),
Rule(re.compile(r"(ய்)\s*([அஆஇஈஉஊஎஏஒஓஐஔ])"), r"\1" + BOUND + r"\2"),
Rule(re.compile(r"(ல்)\s*([அஆஇஈஉஊஎஏஒஓஐஔ])"), r"\1" + BOUND + r"\2"),
Rule(re.compile(r"(ள்)\s*([அஆஇஈஉஊஎஏஒஓஐஔ])"), r"\1" + BOUND + r"\2"),
Rule(re.compile(r"(ர்)\s*([அஆஇஈஉஊஎஏஒஓஐஔ])"), r"\1" + BOUND + r"\2"),

# ---------------------------------------------------------------------
# G) Case-suffix & postposition joins (வேற்றுமைச் சந்தி) — frequent cues
# ---------------------------------------------------------------------

Rule(re.compile(r"([அஆஇஈஉஊஎஏஒஓஐஔ])\s*(ஐ)"), r"\1" + BOUND + r"\2"),
Rule(re.compile(r"([அஆஇஈஉஊஎஏஒஓஐஔ])\s*((உ|க்)கு)"), r"\1" + BOUND + r"\2"),
Rule(re.compile(r"([அஆஇஈஉஊஎஏஒஓஐஔ])\s*(ஆல்|னால்)"), r"\1" + BOUND + r"\2"),
Rule(re.compile(r"([அஆஇஈஉஊஎஏஒஓஐஔ])\s*(இல்|அல்)"), r"\1" + BOUND + r"\2"),
Rule(re.compile(r"([அஆஇஈஉஊஎஏஒஓஐஔ])\s*(இடம்|உடன்|முன்|பின்)"), r"\1" + BOUND + r"\2"),

# Noun + plural/collective markers
Rule(re.compile(r"([அஆஇஈஉஊஎஏஒஓஐஔ])\s*(கள்)"), r"\1" + BOUND + r"\2"),
Rule(re.compile(r"([அஆஇஈஉஊஎஏஒஓஐஔ])\s*(வர்|வர்கள்)"), r"\1" + BOUND + r"\2"),

# ---------------------------------------------------------------------
# H) Verbal participle and auxiliary joins (எச்சம்/வினைச் சந்தி)
# ---------------------------------------------------------------------

# -இ/-உ/-அ participles + போ/வா/இரு/உள்…
Rule(re.compile(r"(ி|உ|அ)\s*(போ|வா|இரு|உள்)"), r"\1" + BOUND + r"\2"),

# -த்து/-ட்டு + auxiliary
Rule(re.compile(r"(த்து|ட்டு)\s*(கொள்|விடு|போ|ஆகு)"), r"\1" + BOUND + r"\2"),

# -ஆன/-என்/-உம் adjectival/relativizer + noun
Rule(re.compile(r"(ஆன|என்|உம்)\s*([அஆஇஈஉஊஎஏஒஓஐஔஅ-ஹ])"), r"\1" + BOUND + r"\2"),

# ---------------------------------------------------------------------
# I) Numeral + classifier/suffix
# ---------------------------------------------------------------------

Rule(re.compile(r"([௦-௯0-9]+)\s*(ஆம்(?: நாள்| ஆண்டு)?)"), r"\1" + BOUND + r"\2"),
Rule(re.compile(r"([௦-௯0-9]+)\s*(ஐ)"), r"\1" + BOUND + r"\2"),

# ---------------------------------------------------------------------
# J) Generic whitespace suppression inside compounds
# ---------------------------------------------------------------------

Rule(re.compile(r"(\S)\s+(?=\S)"), r"\1" + BOUND),
]

# English: pass-through (no phonological sandhi)
EN_RULES: List[Rule] = []

LANG_RULES = {
    "ta": TA_RULES,
    "tamil": TA_RULES,   # alias
    "en": EN_RULES,
    "english": EN_RULES,
}

# ---------- Helpers for code-mixed handling ----------

TAMIL_RANGE = r"\u0B80-\u0BFF"
RE_TAMIL = re.compile(fr"[{TAMIL_RANGE}]")
RE_WORD_OR_SPACE_OR_PUNC = re.compile(r"\w+|\s+|[^\w\s]")

def apply_rules(text: str, rules: List[Rule]) -> str:
    out = text
    for r in rules:
        out = r.pattern.sub(r.repl, out)
    return out

def sandhi_mark(text: str, lang="ta"):
    rules = LANG_RULES.get(lang, [])
    return apply_rules(text, rules)

def _mark_mixed(text: str) -> str:
    """
    Apply Tamil sandhi rules only to Tamil spans; leave non-Tamil spans as-is.
    This ensures English/Tanglish chunks don't get Tamil-specific boundaries.
    """
    chunks = RE_WORD_OR_SPACE_OR_PUNC.findall(text)
    out_parts = []
    for ch in chunks:
        if RE_TAMIL.search(ch):
            out_parts.append(sandhi_mark(ch, "ta"))
        else:
            # English/Latin/digits/punct/spaces -> no sandhi rules
            out_parts.append(sandhi_mark(ch, "en"))  # pass-through
    return "".join(out_parts)

def sandhi_split(text: str, lang="ta") -> List[Tuple[str, Tuple[int,int]]]:
    """
    Returns [(token, (start,end))] splitting on BOUND after applying rules.
    Keeps offsets relative to the *post-rule* string.
    - lang="ta" -> Tamil rules
    - lang="en" -> pass-through
    - lang="mix" -> per-span Tamil-only marking
    """
    if lang.lower() in ("mix", "code-mix", "codemix", "cmix"):
        marked = _mark_mixed(text)
    else:
        marked = sandhi_mark(text, lang)

    parts = marked.split(BOUND)
    tokens = []
    cursor = 0
    for part in parts:
        for w in re.findall(r"\S+|\s+", part):
            tokens.append((w, (cursor, cursor+len(w))))
            cursor += len(w)
    return tokens

def remove_boundaries(text: str) -> str:
    return text.replace(BOUND, "")
