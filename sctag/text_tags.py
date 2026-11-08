"""Genre tags from metadata text: the title, the uploader's genre field and their tags.

Cheap (no audio), so it also covers DRM tracks, and much more reliable than audio zero-shot when it fires.
Titles that name a rapper ("Chief Keef - Citgo (… Blend)") get "rap edit", or "rap" if the rapper uploaded it.
Extend GENRE_PATTERNS / RAPPERS freely: matching is case- and accent-insensitive.
"""

import re
import unicodedata

# tag -> regexes (on normalized text: lowercase, accents stripped, "$" -> "s")
GENRE_PATTERNS = {
    "baile funk": [r"baile ?funk", r"funk carioca", r"funk brasileiro", r"brazilian funk", r"\bfunk\b"],
    "bruxaria": [r"bruxaria"],
    "mandelão": [r"mandelao"],
    "rock doido": [r"rock ?doido"],
    "funk ritmado": [r"ritmad[ao]"],
    "funk automotivo": [r"automotivo"],
    "tecnobrega": [r"tecno ?brega", r"tecno ?melody", r"tecno ?mandelody"],
    "favela bass": [r"favela bass"],
    "jersey club": [r"jersey ?club", r"ジャージー"],
    "deconstructed club": [r"deconstructed", r"decon club"],
    "guaracha / aleteo": [r"guaracha", r"aleteo"],
    "reggaeton / dembow": [r"reggaeton", r"perreo", r"dembow"],
    "latin club": [r"latin ?core", r"latin ?club"],
    "cumbia": [r"cumbia"],
    "kuduro": [r"kuduro"],
    "footwork": [r"footwork", r"\bjuke\b"],
    "grime": [r"\bgrime\b"],
    "drill": [r"\bdrill\b"],
    "rap": [r"\brap\b", r"hip ?-?hop", r"\bplugg\b", r"cloud rap"],
    "trap": [r"\btrap\b"],
    "techno": [r"techno"],
    "hard techno": [r"hard ?techno"],
    "trance": [r"trance"],
    "ambient": [r"ambient"],
    "dubstep / bass": [r"dubstep"],
    "drum & bass / jungle": [r"drum ?(and|&|n) ?bass", r"\bdnb\b", r"\bjungle\b"],
    "uk garage": [r"\bukg\b", r"uk garage", r"2 ?-?step"],
    "breakcore": [r"breakcore"],
    "hardstyle": [r"hardstyle"],
    "hyperpop": [r"hyperpop"],
    "dancehall": [r"dancehall"],
    "house": [r"\bhouse\b"],
    "experimental / noise": [r"experimental", r"\bnoise\b"],
}
TAG_STUFFING = 4  # this many genres in the uploader's tags = keyword stuffing: ignore the tags
BRAZILIAN_FUNK = {"bruxaria", "mandelão", "rock doido", "funk ritmado", "funk automotivo", "tecnobrega", "favela bass"}

# Distinctive names: matched anywhere in the title.
RAPPERS = """
young thug | asap rocky | asap ferg | chief keef | gucci mane | lil wayne | playboi carti | carti | lil uzi vert
travis scott | kanye west | kendrick lamar | baby keem | sexyy red | flo milli | cardi b | ice spice | nicki minaj
megan thee stallion | glorilla | latto | city girls | bhad bhabie | quavo | migos | lil baby | lil durk | gunna
21 savage | metro boomin | yeat | ken carson | destroy lonely | lil yachty | lil tecca | lil tjay | pop smoke
nba youngboy | nba yb | youngboy never broke again | king von | polo g | juice wrld | xxxtentacion | lil peep
ski mask the slump god | denzel curry | rico nasty | doechii | tyler the creator | earl sweatshirt | mf doom
a tribe called quest | wu-tang | ghostface killah | method man | mobb deep | the notorious big | biggie | 2pac
tupac | snoop dogg | dr dre | eminem | 50 cent | lil kim | missy elliott | ludacris | lil jon | three 6 mafia
three six mafia | project pat | juicy j | dj paul | ying yang twins | soulja boy | waka flocka | rick ross
meek mill | jeezy | young jeezy | plies | trina | trick daddy | uncle luke | 2 live crew | ugk | pimp c
bun b | three 6 | dj unk | dem franchize boyz | speaker knockerz | cash out | rich homie quan | lil keed
lil gotit | rx papi | lazer dim 700 | lazerdim700 | esdeekid | fakemink | rico ace | 2hollis | bossman dlow
sheff g | kay flock | fivio foreign | central cee | digga d | skepta | stormzy | giggs | headie one | aitch
slowthai | jme | wiley | dizzee rascal | kano | ghetts | d double e | freeze corleone | pnl | booba | kaaris
niska | ninho | gazo | tiakola | ateyaba | himra | princess nokia | azealia banks | kodak black | ynw melly
mexikodro | lucki | autumn | slayr | nettspend | osamason | che | kankan | summrs | rylo rodriguez
a boogie wit da hoodie | pnb rock | matue | veigh | teto | mc cabelinho | orochi | daboor | shabjdeed | wegz
marwan pablo | bad bunny
""".replace("\n", "|")
# Names that are also ordinary words: only matched as a whole artist credit (e.g. "Future - Honest (… mix)").
AMBIGUOUS_RAPPERS = {
    "future",
    "drake",
    "ye",
    "dave",
    "nas",
    "jt",
    "takeoff",
    "offset",
    "common",
    "logic",
    "jul",
    "sch",
    "eve",
    "russ",
    "tyga",
    "unk",
}


_KANA = re.compile(r"[぀-ヿ]")


def norm(s):
    """Lowercase, accents stripped (Japanese kept as is), "$" -> "s", no apostrophes or dots, single spaces."""
    s = s or ""
    if not _KANA.search(s):
        s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = s.lower().replace("$", "s").replace("'", "").replace(".", "")
    return re.sub(r"\s+", " ", s).strip()


_RAPPER_RE = re.compile(
    r"\b("
    + "|".join(sorted({re.escape(norm(n)) for n in RAPPERS.split("|") if n.strip()}, key=len, reverse=True))
    + r")\b"
)
_GENRE_RE = {tag: re.compile("|".join(pats)) for tag, pats in GENRE_PATTERNS.items()}


def artist_credits(title):
    """Whole artist names credited in a title: segments around ' - ' split on x / & / , / feat."""
    t = re.sub(r"[\(\[\{].*?[\)\]\}]", " ", norm(title))
    parts = []
    for seg in re.split(r"\s[-–—]\s|\s?[-–—]\s|\s[-–—]\s?", t):
        parts += re.split(r"\s*(?:,|&|\+\+|/|\sx\s|\sfeat\s|\sft\s|\swith\s|\svs\s)\s*", seg)
    return {p.strip() for p in parts if p.strip()}


def rappers_in(title):
    """Rapper names found in a title."""
    t = norm(title)
    return set(_RAPPER_RE.findall(t)) | (artist_credits(title) & AMBIGUOUS_RAPPERS)


def text_tags(track, kind="tracks"):
    """[(tag, source)] with source in title / genre / tags; first source wins per tag."""
    found = {}
    for source, text in (
        ("title", track.get("title")),
        ("genre", track.get("sc_genre")),
        ("tags", track.get("sc_tags")),
    ):
        in_text = [tag for tag, rx in _GENRE_RE.items() if rx.search(norm(text))]
        if source == "tags" and len(in_text) >= TAG_STUFFING:  # "Dembow Reggaetón Hardstyle Techno ..." = SEO
            continue
        for tag in in_text:
            found.setdefault(tag, source)
    if found.keys() & BRAZILIAN_FUNK:
        found.setdefault("baile funk", next(found[g] for g in found if g in BRAZILIAN_FUNK))
    if kind == "tracks" and (names := rappers_in(track.get("title"))):
        uploader = re.sub(r"[^a-z0-9]", "", norm(track.get("artist")))
        own_upload = any(re.sub(r"[^a-z0-9]", "", n) == uploader for n in names)
        found["rap" if own_upload else "rap edit"] = "title"
    return list(found.items())
