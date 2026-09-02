"""Language registry for the multilingual cross-script entity-matching corpus.

Each entry names one language, the script its entities are natively written in, the
Wikidata language code used to fetch labels, and the country QIDs used to find entities
native to that language's region.

Two kinds of contrast are supported.

  NON_LATIN languages vary *script*: the native form is in a non-Latin script and the
  comparison form is the canonical Latin spelling.

  LATIN languages vary *canonical form* with script held constant: the native form
  carries diacritics and the comparison form has them stripped. This isolates the
  canonical-form account from the script account, which no non-Latin language can do.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Language:
    key: str                    # short handle used in filenames
    name: str                   # human-readable
    wikidata_lang: str          # label language code on Wikidata
    script: str                 # script name
    script_ranges: tuple        # (lo, hi) Unicode codepoint ranges defining the script
    country_qids: tuple         # for finding entities native to this language's region
    family: str                 # grouping, for reporting
    contrast: str = "script"    # "script" or "diacritic"
    notes: str = ""

    def in_script(self, text: str) -> bool:
        """True if the text contains at least one character of this script."""
        return any(any(lo <= ord(ch) <= hi for lo, hi in self.script_ranges) for ch in text)

    def script_purity(self, text: str) -> bool:
        """True if the text contains no Latin letters (for non-Latin scripts)."""
        if self.contrast == "diacritic":
            return True
        return not any("A" <= ch <= "Z" or "a" <= ch <= "z" for ch in text)


LATIN = ((0x0041, 0x005A), (0x0061, 0x007A), (0x00C0, 0x024F))

# Country QIDs must include the historical and successor states, not only the modern
# country. Wikidata records citizenship as it stood: most notable Russians carry the Soviet
# Union or the Russian Empire rather than Q159, most Serbs carry one of the Yugoslavias,
# most Turks before 1923 carry the Ottoman Empire, and most Indians before 1947 carry the
# British Raj. The first run of 01_entity_pool.py listed only the modern states and found
# 25 native Russians, 13 Arabs, 17 Chinese and 7 Serbs in pools of several hundred, which
# is a property of these lists rather than of Wikidata.
#
# Run verify_country_qids.py after editing: it resolves every QID here against Wikidata and
# prints its English label, so a transposed digit shows up as a country nobody meant.


# --------------------------------------------------------------------------------------
# Non-Latin scripts: the script contrast
# --------------------------------------------------------------------------------------
_NON_LATIN = [
    # Devanagari, three languages sharing one script
    Language("mr", "Marathi",  "mr", "Devanagari", ((0x0900, 0x097F),), ("Q668", "Q129286"),  "Indo-Aryan",
             notes="the original corpus language"),
    Language("hi", "Hindi",    "hi", "Devanagari", ((0x0900, 0x097F),), ("Q668", "Q129286"),  "Indo-Aryan"),
    Language("ne", "Nepali",   "ne", "Devanagari", ((0x0900, 0x097F),), ("Q837",),  "Indo-Aryan"),
    # Sanskrit shares Devanagari with Marathi and Hindi, which makes the three of them the
    # sharpest test the corpus can run: one script, three languages. If the deficit is
    # about the script, all three should show it; if it tracks how much text the model saw
    # in the language, they should separate, since Sanskrit is by far the thinnest.
    #
    # Two caveats. It is a classical language, so the native-versus-foreign origin split is
    # strained: its entities are ancient figures whose citizenship Wikidata often records as
    # a kingdom or not at all, so expect few natives regardless of the country list. And it
    # is low-resource on Wikidata, so it may not support a full corpus at all; check the
    # pool size before building anything on it.
    Language("sa", "Sanskrit", "sa", "Devanagari", ((0x0900, 0x097F),), ("Q668", "Q129286"),
             "Indo-Aryan",
             notes="classical; shares Devanagari with mr and hi, so isolates language from script"),

    # Dravidian, four distinct scripts in one family
    Language("ta", "Tamil",     "ta", "Tamil",     ((0x0B80, 0x0BFF),), ("Q668", "Q854", "Q129286"), "Dravidian"),
    Language("te", "Telugu",    "te", "Telugu",    ((0x0C00, 0x0C7F),), ("Q668", "Q129286"), "Dravidian"),
    Language("kn", "Kannada",   "kn", "Kannada",   ((0x0C80, 0x0CFF),), ("Q668", "Q129286"), "Dravidian"),
    Language("ml", "Malayalam", "ml", "Malayalam", ((0x0D00, 0x0D7F),), ("Q668", "Q129286"), "Dravidian"),

    # Other Indic scripts
    Language("bn", "Bengali",   "bn", "Bengali",   ((0x0980, 0x09FF),), ("Q668", "Q902", "Q129286"), "Indo-Aryan"),
    Language("gu", "Gujarati",  "gu", "Gujarati",  ((0x0A80, 0x0AFF),), ("Q668", "Q129286"), "Indo-Aryan"),
    Language("pa", "Punjabi",   "pa", "Gurmukhi",  ((0x0A00, 0x0A7F),), ("Q668", "Q129286", "Q843"), "Indo-Aryan"),

    # High-resource non-Latin
    Language("ru", "Russian",  "ru", "Cyrillic", ((0x0400, 0x04FF),), ("Q159", "Q15180", "Q34266", "Q2184"), "Slavic"),
    Language("ar", "Arabic",   "ar", "Arabic",   ((0x0600, 0x06FF),), ("Q851", "Q79", "Q796", "Q858", "Q822", "Q1028", "Q262",
                                                  "Q948", "Q878", "Q810", "Q219060", "Q817", "Q1049",
                                                  "Q1016", "Q805", "Q842", "Q846", "Q398", "Q1025"), "Semitic"),
    Language("zh", "Chinese",  "zh", "Han",      ((0x4E00, 0x9FFF),), ("Q148", "Q865", "Q8646", "Q14773", "Q13426199", "Q8733"), "Sinitic"),
    Language("ko", "Korean",   "ko", "Hangul",   ((0xAC00, 0xD7AF), (0x1100, 0x11FF)), ("Q884", "Q423", "Q28179", "Q28233"), "Koreanic"),
    Language("el", "Greek",    "el", "Greek",    ((0x0370, 0x03FF),), ("Q41",),  "Hellenic"),
    Language("he", "Hebrew",   "he", "Hebrew",   ((0x0590, 0x05FF),), ("Q801", "Q193714"), "Semitic"),
    Language("th", "Thai",     "th", "Thai",     ((0x0E00, 0x0E7F),), ("Q869",), "Tai-Kadai"),

    # Japanese: the falsification case. Katakana is a dedicated script for foreign names,
    # so the native rendering of a foreign entity IS canonical here, unlike Devanagari.
    Language("ja", "Japanese", "ja", "Kana/Kanji",
             ((0x3040, 0x309F), (0x30A0, 0x30FF), (0x4E00, 0x9FFF)), ("Q17", "Q188712"), "Japonic",
             notes="Katakana is the canonical script for foreign names; predicts a smaller gap"),

    # Digraphic controls: one language, two scripts
    Language("sr", "Serbian", "sr", "Cyrillic", ((0x0400, 0x04FF),), ("Q403", "Q36704", "Q83286", "Q37024", "Q191077"), "Slavic",
             notes="natively digraphic; Latin form is also native, not a transliteration"),
    Language("ur", "Urdu",    "ur", "Arabic",   ((0x0600, 0x06FF),), ("Q843", "Q129286", "Q668"), "Indo-Aryan",
             notes="pairs with Hindi: one spoken language, two scripts"),
]

# --------------------------------------------------------------------------------------
# Latin scripts: the diacritic contrast, script held constant
# --------------------------------------------------------------------------------------
_LATIN = [
    # German and French: Latin script with a moderate diacritic load, and unlike the
    # earlier Latin entries their *native* figures reliably carry the marks -- umlauts and
    # the eszett, accents and the cedilla -- so the diacritic filter should not gut them the
    # way it gutted Vietnamese. Both are also high-resource, which makes them the cleanest
    # available test of whether the deficit is about script at all: if stripping diacritics
    # from a German name costs accuracy, the effect was never about writing systems.
    Language("de", "German", "de", "Latin", LATIN,
             ("Q183", "Q40", "Q39", "Q43287", "Q41304", "Q7318", "Q16957", "Q713750",
              "Q38872"), "Germanic", "diacritic",
             notes="umlauts and eszett; high-resource, so a null here is informative"),
    Language("fr", "French", "fr", "Latin", LATIN,
             ("Q142", "Q31", "Q39", "Q70802", "Q70972", "Q71084"), "Romance", "diacritic",
             notes="accents and cedilla; high-resource"),

    Language("vi", "Vietnamese", "vi", "Latin", LATIN, ("Q881", "Q180573", "Q172640", "Q185682"), "Austroasiatic", "diacritic",
             notes="heaviest diacritic load; the sharpest canonical-form test"),
    Language("tr", "Turkish",    "tr", "Latin", LATIN, ("Q43", "Q12560"),  "Turkic",        "diacritic",
             notes="dotted and dotless i are distinct letters"),
    Language("cs", "Czech",      "cs", "Latin", LATIN, ("Q213", "Q33946"), "Slavic",        "diacritic"),
    Language("yo", "Yoruba",     "yo", "Latin", LATIN, ("Q1033", "Q962"), "Niger-Congo",  "diacritic",
             notes="tone marks; low resource"),
    Language("is", "Icelandic",  "is", "Latin", LATIN, ("Q189",), "Germanic",      "diacritic",
             notes="thorn and eth are not ASCII-representable"),

    # Near-null controls: Latin script, minimal diacritics, so little should change
    Language("id", "Indonesian", "id", "Latin", LATIN, ("Q252", "Q188161"), "Austronesian",  "diacritic",
             notes="near-null control"),
    Language("sw", "Swahili",    "sw", "Latin", LATIN, ("Q924", "Q114", "Q1036", "Q974"), "Niger-Congo", "diacritic",
             notes="near-null control"),
]

LANGUAGES = {lang.key: lang for lang in (_NON_LATIN + _LATIN)}

NON_LATIN_KEYS = tuple(l.key for l in _NON_LATIN)
LATIN_KEYS = tuple(l.key for l in _LATIN)

# --------------------------------------------------------------------------------------
# The set to run
# --------------------------------------------------------------------------------------
# Two halves that answer different questions, which is why the Latin half has to be as
# populated as the non-Latin one rather than carrying a single token member.
#
# The non-Latin half asks whether a change of script costs the model access to an entity.
# Within it, mr/hi/sa hold Devanagari constant while the language varies, which is the only
# way to separate a script effect from a language-resource effect; ja and sr are the two
# cells that can damage the account, since katakana is the canonical script for foreign
# names and Serbian's Latin form is native rather than a transliteration.
#
# The Latin half holds script constant and varies only the canonical written form, by
# stripping diacritics. That is the control the script account needs: if accuracy falls
# here too, the effect was never about script but about matching the form the entity was
# learned under. A single Latin language cannot carry that, because a null result would be
# indistinguishable from that language simply being easy. Several, spanning diacritic load
# from heavy to near-absent, give it a dose-response to fail or satisfy.
# --------------------------------------------------------------------------------------
# RUN_10: the set chosen after fetching, on what the pools actually contain
# --------------------------------------------------------------------------------------
# The fetch settled a question the design could not. The origin contrast needs entities
# native to the language, and only five languages have enough of them:
#
#     hi 150, ta 150, yo 150, mr 127, sa 116     then te 74, ru 33, and everything else
#     under 30, down to vi at zero.
#
# The reason is the corpus definition, not the country lists. An item needs a native alias
# and an English alias, which selects for people famous in two language communities at
# once. For Indic languages and Yoruba, local figures clear that bar; for Vietnamese or
# Czech the pool fills with international celebrities instead. So the native-versus-foreign
# prediction can only be tested where it is testable, and the rest carry the script
# contrast alone. Both halves are chosen deliberately below.
RUN_9 = (
    # Origin contrast testable: both native and foreign present in useful numbers.
    "mr",   # 724 items, the paperB corpus
    "hi",   # 365
    "sa",   # 171, and the most balanced pool of all: 96 native against 75 foreign
    "ta",   # 322
    # Script contrast only: too few natives to test origin, but each covers a
    # writing-system property nothing else in the set does.
    "ru",   # 236. Cyrillic, high-resource.
    "ar",   # 324. Right-to-left, and an abjad rather than an alphabet.
    "ja",   # 318. Katakana is canonical for foreign names, so the gap should shrink here.
    "sr",   # 233. Natively digraphic: the Latin form is also native.
    "zh",   # 312. Logographic, the remaining writing-system type.
)

# Vietnamese and Yoruba were dropped after the fetch. Their pools are dominated by plain
# ASCII names of international figures, so once the diacritic filter is applied only twelve
# and seventeen percent survive, leaving 35 and 75 items. Both are too small to match the
# others, and Yoruba additionally loses its origin contrast to the same filter: 150 native
# entities before it, one after, because Yoruba names carrying tone marks are overwhelmingly
# the local figures.
#
# Their departure costs the design its whole Latin half, and with it the control that holds
# script constant while varying only the canonical written form. What remains of that
# question is stage 12, which romanizes the native names and asks the same thing within a
# language rather than across a set of them. If a Latin cell is wanted back later, Icelandic
# is the strongest candidate: it had the best diacritic yield of any Latin language at
# twenty-three percent, and thorn and eth have no ASCII form at all.
DROPPED_AFTER_FETCH = ("vi", "yo", "te", "tr", "cs", "is", "id", "sw")

DEVANAGARI_TRIO = ("mr", "hi", "sa")

# Where the native-versus-foreign prediction can actually be tested.
ORIGIN_TESTABLE = ("mr", "hi", "sa", "ta", "yo")

NON_LATIN_RUN = ("mr", "hi", "sa",      # one script, three languages
                 "ta",                  # Dravidian, own script, fetched cleanly at 450
                 "ru", "ar", "zh",      # high-resource, three unrelated scripts
                 "ja",                  # katakana is canonical for foreign names
                 "sr")                  # digraphic: the Latin form is also native

LATIN_RUN = ("vi",   # heaviest diacritic load, the sharpest canonical-form test
             "tr",   # dotted and dotless i are distinct letters, not decoration
             "cs",   # pairs with ru and sr: three Slavic languages, two scripts
             "is",   # thorn and eth have no ASCII form at all
             "yo",   # tone marks, and low-resource, so it tests both at once
             "id",   # near-null control: little to strip, so little should change
             "sw")   # second near-null control, unrelated family

RUN_SET = NON_LATIN_RUN + LATIN_RUN

# Thin on Wikidata: these time the query service out at the usual settings and need a low
# --per-band, or may not support a full corpus at all. Kept separate so a run of RUN_SET is
# not held up by them.
THIN = ("sa", "te", "yo")

# The earlier twelve, kept so that run can be reproduced.
PILOT_SET = ("mr", "hi", "ta", "te", "ru", "ar", "zh", "ja", "sr", "vi", "tr", "id")


def by_family() -> dict:
    out: dict = {}
    for lang in LANGUAGES.values():
        out.setdefault(lang.family, []).append(lang.key)
    return out


if __name__ == "__main__":
    print(f"{len(LANGUAGES)} languages registered\n")
    print(f"{'key':<5}{'name':<12}{'script':<12}{'contrast':<11}{'family'}")
    for lang in LANGUAGES.values():
        print(f"{lang.key:<5}{lang.name:<12}{lang.script:<12}{lang.contrast:<11}{lang.family}")
    print(f"\nnon-Latin: {len(NON_LATIN_KEYS)} | Latin: {len(LATIN_KEYS)}")
    print(f"pilot set: {len(PILOT_SET)} -> {', '.join(PILOT_SET)}")
