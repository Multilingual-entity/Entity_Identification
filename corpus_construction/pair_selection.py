# SUPERSEDED. Measured, found insufficient, and not used to build the shipped corpora.
#
# This selects the cross-script name pair by transliteration distance and refuses the
# entity when no pair scores well. Checked against human labels it reached 59 percent
# precision structurally and 68 percent component-wise, so roughly a third of what it
# accepted was wrong and some of what it refused was fine. Automatic cross-script name
# matching topped out near 75 percent agreement in every variant tried, which is why
# the shipped corpora were verified by a reader instead.
#
# Still imported by 01_entity_pool.py for its scoring helpers, which are sound; it is
# acting on the score unconditionally that fails.

r"""Choose the cross-script name pair, and refuse the entity when no good pair exists.

best_matching_alias already scores every English alias against the native one, which was
the right idea. What it does not do is act on the score: it returns the best candidate
unconditionally, so an entity whose aliases simply do not correspond is kept anyway with
its least-bad pairing. That is where most of the Hindi corpus went wrong -- 51 rows holding
two genuinely different names, and 67 more where one script carries a name component the
other lacks. Both fields are correct about the person and wrong as a pair, so the Devanagari
prompt and the Latin prompt ask different questions and the script contrast is confounded.

Two things have to be true of this module or it makes the corpus worse rather than better,
and an earlier attempt failed both:

  1. It must not decide by a similarity score alone. anyascii romanizes Devanagari crudely,
     so a low score does not mean a mismatch: फ्लोरेंस ईजेकील / Florence Ezekiel and
     कमलजीत सिंह झूटी / Kamaljit Singh Jhooti both score badly and are both correct. Every
     rejection here is instead structural -- a bracket, a joined list, a token with no
     counterpart -- and structure survives a bad transliteration.

  2. It must not maximise similarity when picking among candidates. Short strings score
     higher, so plain argmax replaces हरिवंश राय श्रीवत्सव / Harivansh Rai Shrivastav with
     बच्चन / Bachchan, and a full name is lost to a surname. Candidates are therefore ranked
     by whether their components correspond first, and by similarity only to break ties.

The similarity score is still computed and returned, but only to order the review sheet.
No entity is dropped for scoring low; entities are dropped for being structurally broken,
and the reader decides the rest.
"""
from __future__ import annotations

import difflib
import re
import unicodedata

# Devanagari letters, the danda, and the Latin block. Anything else in a native field --
# a bracket, a pipe, a backslash -- means the field is not a plain name.
_DEV = re.compile(r"[ऀ-ॿ]")
_LATIN = re.compile(r"[A-Za-z]")

# Stripped before components are compared. An honorific may sit on one side of a pair
# without making the two names different: "Pandit Ravi Shankar" and "रवि शंकर" are the
# same name, "Antonio Francesco Gramsci" and "ग्राम्शी" are not.
HONORIFICS_LATIN = {
    "sir", "dr", "mr", "mrs", "ms", "shri", "sri", "smt", "swami", "pandit", "pt",
    "professor", "prof", "general", "gen", "colonel", "captain", "lieutenant", "imam",
    "guru", "ustad", "usthad", "ayya", "bhikkhu", "bhikkhuni", "mahathera", "lord", "lady",
    "queen", "king", "emperor", "empress", "pope", "saint", "st", "nawab", "sheikh",
    "shaikh", "hazrat", "acharya", "maulana", "rt", "hon", "blessed", "the", "of", "ji",
    # Nobiliary and patronymic particles. Lowercase by convention, not by carelessness,
    # so they must be exempt from the capitalisation check as well as from the components.
    "do", "da", "de", "del", "della", "di", "du", "dos", "van", "von", "der", "den",
    "bin", "ibn", "al", "el", "la", "le", "ter", "ten",
}
HONORIFICS_NATIVE = {
    "सर", "डॉ", "डा", "श्री", "श्रीमती", "स्वामी", "पंडित", "पण्डित", "प्रोफेसर", "जनरल",
    "कर्नल", "कप्तान", "इमाम", "गुरु", "उस्ताद", "आचार्य", "मौलाना", "लॉर्ड", "लार्ड",
    "सम्राट", "महारानी", "राजकुमार", "पोप", "नवाब", "शेख", "हज़रत", "हजरत", "भिक्खू",
    "भिक्खु", "भिक्षुणी", "माननीय", "जी", "संत", "दा", "दे", "ला", "ले",
    "वान", "फ़ॉन", "फॉन", "अल", "इब्न", "बिन", "आपा", "साहब", "साहेब",
}


def romanizations(text: str) -> list:
    """Several Latin readings of one native string. A name only has to look right under
    one of them: ITRANS writes the inherent vowel where IAST does not, and Aksharamukha
    handles conjuncts the general-purpose mappers turn to mush."""
    out, text = [], str(text)
    for module, fn in (("anyascii", "anyascii"), ("unidecode", "unidecode")):
        try:
            out.append(getattr(__import__(module), fn)(text))
        except Exception:                                                    # noqa: BLE001
            pass
    try:
        from indic_transliteration import sanscript
        from indic_transliteration.sanscript import transliterate
        for scheme in (sanscript.IAST, sanscript.ITRANS, sanscript.HK):
            try:
                out.append(transliterate(text, sanscript.DEVANAGARI, scheme))
            except Exception:                                                # noqa: BLE001
                pass
    except ImportError:
        pass
    try:
        from aksharamukha import transliterate as aksh
        out.append(aksh.process("Devanagari", "ISO", text))
    except Exception:                                                        # noqa: BLE001
        pass
    return [r for r in out if r] or [text]


def fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", str(text))
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return "".join(c.lower() for c in stripped if c.isalnum())


def fold_native(text: str) -> str:
    """Combining marks kept. Indic vowel signs are combining characters, and dropping
    them collapses unrelated words onto each other."""
    return "".join(c.lower() for c in unicodedata.normalize("NFC", str(text)) if c.isalnum())


def tokens(text: str, native: bool) -> list:
    """Name components. Honorifics and bare initials are not components: they may differ
    between the two scripts without the names differing."""
    parts = re.split(r"[\s.,;:।\"'()\[\]|/\\-]+", str(text))
    drop = HONORIFICS_NATIVE if native else HONORIFICS_LATIN
    return [p for p in parts if p and p.lower() not in drop and len(p) > 1]


def similarity(native: str, latin: str) -> float:
    """Best agreement between any reading of the native string and the Latin one."""
    target = fold(latin)
    if not target:
        return 0.0
    best = 0.0
    for reading in romanizations(native):
        folded = fold(reading)
        if folded:
            best = max(best, difflib.SequenceMatcher(None, folded, target).ratio())
    return best


def uncovered_tokens(native: str, latin: str) -> tuple:
    """Latin components with no counterpart in the native string, and the reverse.

    Token counts alone cannot do this: Devanagari writes शाहरुख़ ख़ान where English writes
    Shah Rukh Khan, so the counts differ while every component is present. What matters is
    whether each component appears at all, which is asked by looking for it inside the
    romanized whole rather than by lining tokens up one to one.
    """
    # The romanizations keep their spacing, so this is a comparison between component and
    # component rather than a search through a run of characters. That matters: राय reads
    # as "ray" or "raaya" depending on the scheme, and only a token-level comparison
    # recognises those as the same component as "Rai".
    native_components = [
        [fold(t) for t in reading.split()] for reading in romanizations(native)
    ]
    latin_components = [fold(t) for t in tokens(latin, native=False)]

    missing_from_native = [
        t for t in tokens(latin, native=False)
        if not any(_present(fold(t), c) for reading in native_components for c in reading)
    ]
    extra_in_native = [
        t for t in tokens(native, native=True)
        if not any(_present(fold(r), c)
                   for r in romanizations(t) for c in latin_components)
    ]
    return missing_from_native, extra_in_native


# Distinctions the Devanagari-to-Latin schemes disagree about. ज़ comes back as j or z,
# व as v or w, फ़ as ph or f, and the schemes differ on vowel length throughout. Two
# spellings that differ only in these ways are the same component.
_EQUIVALENCES = [("sh", "s"), ("zh", "s"), ("ch", "c"), ("kh", "k"), ("gh", "g"),
                 ("th", "t"), ("dh", "d"), ("bh", "b"), ("ph", "f"), ("z", "j"),
                 ("w", "v"), ("y", "i"), ("q", "k"), ("x", "ks"), ("aa", "a"),
                 ("ee", "i"), ("ii", "i"), ("oo", "u"), ("uu", "u")]


def _skeleton(text: str) -> str:
    """The consonants that survive any transliteration scheme."""
    out = text
    for a, b in _EQUIVALENCES:
        out = out.replace(a, b)
    return "".join(c for c in out if c not in "aeiou") or out


def _present(needle: str, candidate: str, threshold: float = 0.62) -> bool:
    """Are these the same name component, allowing for a rough transliteration.

    The threshold is low on purpose, and there is a consonant-skeleton fallback behind it.
    A missed match drops a valid entity, which is the expensive mistake; a spurious match
    only leaves one defect for the reader to catch.
    """
    if not needle or not candidate:
        return False
    # Containment, but only between strings of comparable length. Without the length
    # guard "Ramachandran" would count as covered by राम, and a two-thirds missing
    # component would pass as present.
    if (needle in candidate or candidate in needle) \
            and min(len(needle), len(candidate)) / max(len(needle), len(candidate)) >= 0.6:
        return True
    if difflib.SequenceMatcher(None, needle, candidate).ratio() >= threshold:
        return True
    a, b = _skeleton(needle), _skeleton(candidate)
    return bool(a) and bool(b) and \
        difflib.SequenceMatcher(None, a, b).ratio() >= 0.80


def _has_initials(text: str) -> bool:
    """Does this field abbreviate part of the name? Then the other side may spell out
    components this one does not have, and that is not a missing component."""
    parts = re.split(r"[\s]+", str(text).strip())
    return any(re.fullmatch(r"[^\W\d_]{1,2}[.॰]?", p) and (p.endswith(".")
               or p.endswith("॰") or len(p.rstrip(".॰")) == 1)
               for p in parts if p)


def structural_defects(native: str, latin: str) -> list:
    """Defects that hold whatever the transliteration quality. These are safe to act on."""
    native, latin = str(native), str(latin)
    found = []
    if re.search(r"[(\[]", native) or re.search(r"[(\[]", latin):
        found.append("parenthetical disambiguator")
    if "|" in native or "|" in latin:
        found.append("several aliases in one field")
    if re.search(r",|;|\bया\b|\bकिंवा\b|\bअथवा\b", native):
        found.append("several aliases in one field")
    if "\\" in native or "\\" in latin:
        found.append("stray escape character")
    if _LATIN.search(native):
        found.append("Latin letters in the native field")
    if not _DEV.search(native):
        found.append("native field is not in the native script")
    # Not "all lowercase": the fields that come out of a careless edit are mixed, as in
    # "Johan kristopher hansen", where only the first word was ever capitalised.
    words = [w for w in latin.split() if w and w.lower() not in HONORIFICS_LATIN]
    if len(words) > 1 and any(w[0].isupper() for w in words) \
            and any(w[0].islower() and w[0].isalpha() for w in words):
        found.append("Latin field is inconsistently capitalised")
    missing, extra = uncovered_tokens(native, latin)
    if _has_initials(latin):
        extra = []
    if _has_initials(native):
        missing = []
    if missing:
        found.append(f"the Latin carries {' '.join(missing)}, the native does not")
    if extra:
        found.append(f"the native carries {' '.join(extra)}, the Latin does not")
    return found


def choose_pair(native_aliases, english_aliases, native_label, english_label):
    """The best (native alias, English alias) pairing for one entity, or None.

    Ranked by whether the components correspond, then by similarity. That order is what
    keeps a full name from being replaced by a surname: बच्चन / Bachchan scores higher than
    हरिवंश राय श्रीवत्सव / Harivansh Rai Shrivastav, but only the second has every component
    of its partner, so only the second can win.
    """
    natives = [a for a in dict.fromkeys(native_aliases or [])
               if fold_native(a) != fold_native(native_label)]
    englishes = [a for a in dict.fromkeys(english_aliases or [])
                 if fold(a) != fold(english_label)]
    if not natives or not englishes:
        return None

    best = None
    for native in natives:
        for english in englishes:
            defects = structural_defects(native, english)
            score = similarity(native, english)
            # Fewest defects first, then the fuller name, and only then similarity. The
            # middle term is what stops the collapse to a surname: बच्चन / Bachchan and
            # हरिवंश राय श्रीवत्सव / Harivansh Rai Shrivastav are both defect-free, and the
            # short one scores higher, so similarity alone would discard the full name.
            rank = (-len(defects), len(tokens(english, native=False)), score)
            if best is None or rank > best[0]:
                best = (rank, native, english, score, defects)
    _, native, english, score, defects = best
    return {"native": native, "english": english, "score": score, "defects": defects}
