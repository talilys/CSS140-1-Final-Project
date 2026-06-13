"""
normalizer.py
EaseAI - Pre-processing layer for slangs, emoticons, and sarcasm.

Pipeline (runs BEFORE the ML model):
  1. Emoji → sentiment words
  2. Emoticon faces → sentiment words  
  3. Filipino + Gen Z slang → plain English equivalents
  4. Sarcasm detection → returns flag + sarcasm type

Nothing here touches the model itself — it enriches the raw text
so the TF-IDF vectorizer sees meaningful words.
"""

import re

# ── 1. Emoji → sentiment word mappings ──────────────────────────────────────
# Maps emoji characters to descriptive words the model understands.
EMOJI_MAP = {
    # Negative / stress
    "😭": "crying sad devastated",
    "😢": "sad crying upset",
    "😔": "sad disappointed down",
    "😞": "disappointed sad",
    "😟": "worried stressed",
    "😩": "exhausted overwhelmed stressed",
    "😫": "tired exhausted drained",
    "😤": "frustrated angry annoyed",
    "😠": "angry frustrated mad",
    "😡": "angry furious mad",
    "🤬": "furious angry",
    "😰": "anxious nervous worried",
    "😨": "scared fearful anxious",
    "😱": "shocked scared overwhelmed",
    "🥺": "sad pleading upset",
    "😿": "sad crying",
    "💔": "heartbroken sad hurt",
    "🫠": "melting overwhelmed stressed done",
    "😵": "dizzy overwhelmed lost",
    "😵‍💫": "dizzy confused overwhelmed",
    "🤯": "mind blown overwhelmed shocked",
    "🙃": "frustrated not okay sarcastic",  # sarcasm signal
    "😐": "blank numb okay neutral",
    "😑": "done annoyed deadpan",
    "🥹": "emotional touched trying not to cry",

    # Positive / happy
    "😊": "happy smiling content",
    "😄": "happy excited joyful",
    "😁": "very happy excited thrilled",
    "😆": "laughing happy excited",
    "🥰": "loved grateful happy",
    "😍": "love excited amazing",
    "🤩": "starstruck excited thrilled",
    "😎": "cool confident great",
    "🤗": "happy warm grateful",
    "😇": "blessed thankful happy",
    "🥳": "celebrating excited happy",
    "😂": "funny laughing",
    "🤣": "laughing funny hilarious",
    "✨": "great amazing wonderful",
    "🎉": "celebrating excited happy",
    "🙌": "celebrating proud happy",
    "💪": "strong motivated confident",
    "🔥": "amazing excellent crushing it",
    "⭐": "great excellent proud",
    "💯": "perfect excellent accomplished",
    "🏆": "winning accomplished proud",
    "👏": "well done proud accomplished",
    "💚": "grateful happy loving",
    "💙": "calm peaceful okay",
    "❤️": "love happy grateful",
    "🫶": "grateful caring warm",

    # Exhaustion / burnout (negative)
    "💀": "dead exhausted destroyed done",
    "☠️": "dead exhausted overwhelmed",
    "🪦": "dead done finished giving up",
    "😴": "tired sleepy exhausted",
    "🛌": "exhausted need rest tired",
    "😓": "struggling tired stressed",
    "😥": "sad struggling disappointed",

    # Neutral / thinking
    "🤔": "thinking wondering confused",
    "🙄": "annoyed unimpressed eye roll",  # sarcasm signal
    "😶": "speechless nothing to say",
    "🫤": "unsure mixed feelings not sure",
    "😬": "awkward anxious uncomfortable",
    "🤷": "unsure not sure confused",
    "🤦": "frustrated facepalm annoyed",
}

# ── 2. Text emoticons → sentiment words ─────────────────────────────────────
EMOTICON_MAP = [
    (r"(?<!\w):\)\)*(?!\w)",   "happy smiling"),
    (r"(?<!\w):-\)(?!\w)",     "happy smiling"),
    (r"(?<!\w):D(?!\w)",       "very happy laughing"),
    (r"(?<!\w)=\)(?!\w)",      "happy content"),
    (r"(?<!\w);\)(?!\w)",      "playful winking"),
    (r"(?<!\w):\|(?!\w)",      "neutral expressionless"),
    (r"(?<!\w):-\|(?!\w)",     "neutral deadpan"),
    (r"(?<!\w):\((?!\w)",      "sad unhappy"),
    (r"(?<!\w):-\((?!\w)",     "sad upset"),
    (r"(?<!\w):'\((?!\w)",     "crying sad"),
    (r"(?<!\w):\[\s*(?!\w)",   "sad disappointed"),
    (r"(?<!\w)>:\((?!\w)",     "angry frustrated"),
    (r"(?<!\w):o(?!\w)",       "shocked surprised"),
    (r"(?<!\w):-o(?!\w)",      "shocked overwhelmed"),
    (r"(?<!\w);\((?!\w)",      "crying sad"),
    (r"(?<!\w)T_T(?!\w)",      "crying devastated sad"),
    (r"(?<!\w)TT(?!\w)",       "crying sad"),
    (r"(?<!\w)T\.T(?!\w)",     "crying sad"),
    (r"(?<!\w)QQ(?!\w)",       "crying sad upset"),
    (r"(?<!\w)Q\.Q(?!\w)",     "crying sad"),
    (r"(?<!\w)>_<(?!\w)",      "frustrated annoyed stressed"),
    (r"(?<!\w)-_-(?!\w)",      "done tired deadpan"),
    (r"(?<!\w)\^_\^(?!\w)",    "happy content smiling"),
    (r"(?<!\w)\^~\^(?!\w)",    "happy content"),
    (r"(?<!\w)\*_\*(?!\w)",    "amazed excited starstruck"),
    (r"(?<!\w)u_u(?!\w)",      "sad defeated down"),
    (r"(?<!\w)UwU(?!\w)",      "happy warm content"),
    (r"(?<!\w)OwO(?!\w)",      "surprised excited"),
    (r"(?<!\w):\*/i",          "happy excited"),
    (r"xD(?!\w)",              "laughing very happy"),
    (r"(?<!\w)x3(?!\w)",       "happy cute"),
    (r"(?<!\w)<3(?!\w)",       "love happy grateful"),
    (r"(?<!\w)</3(?!\w)",      "heartbroken sad"),
    (r"(?<!\w>):3(?!\w)",      "cute happy playful"),
    (r"(?<!\w):P(?!\w)",       "playful joking"),
    (r"(?<!\w):-P(?!\w)",      "playful joking"),
    (r"(?<!\w)\(:(?!\w)",      "sad upset"),
    (r"(?<!\w)\)':(?!\w)",     "sad upset crying"),
    (r"(?<!\w)lol(?!\w)",      "laughing funny"),
    (r"(?<!\w)lmao(?!\w)",     "laughing very funny"),
    (r"(?<!\w)lmfao(?!\w)",    "laughing hilarious"),
    (r"(?<!\w)haha(?!\w)",     "laughing happy"),
    (r"(?<!\w)hehe(?!\w)",     "laughing playful"),
    (r"(?<!\w)huhu(?!\w)",     "crying sad upset"),
    (r"(?<!\w)huhuhu(?!\w)",   "crying sad very upset"),
    (r"(?<!\w)hays?(?!\w)",    "disappointed tired resigned"),
    (r"(?<!\w)hayst(?!\w)",    "disappointed tired resigned"),
    (r"(?<!\w)ayy?(?!\w)",     "disappointment sigh"),
    (r"(?<!\w)omg(?!\w)",      "shocked overwhelmed surprised"),
    (r"(?<!\w)omfg(?!\w)",     "shocked very overwhelmed"),
    (r"(?<!\w)ugh(?!\w)",      "frustrated annoyed"),
    (r"(?<!\w)ughhh+(?!\w)",   "very frustrated exhausted"),
    (r"(?<!\w)argh+(?!\w)",    "frustrated angry"),
    (r"(?<!\w)noooo+(?!\w)",   "devastated upset"),
    (r"(?<!\w)yesss+(?!\w)",   "excited very happy"),
    (r"(?<!\w)yay+(?!\w)",     "excited happy celebrating"),
]

# ── 3. Slang → plain English ─────────────────────────────────────────────────
# Filipino student slangs + Gen Z terms, ordered longest-first to avoid
# partial matches (e.g. "grabe" before "gra").
SLANG_MAP = [
    # Filipino slangs
    ("naubos na lakas ko",    "completely exhausted no energy left"),
    ("naubos na ako",         "completely exhausted drained"),
    ("wala na akong pag-asa", "hopeless giving up"),
    ("di ko na kaya",         "cannot handle it anymore overwhelmed"),
    ("hindi ko na kaya",      "cannot handle it anymore overwhelmed"),
    ("sobrang dami",          "so many too much overwhelming"),
    ("sobrang hirap",         "very difficult struggling hard"),
    ("sobrang stress",        "very stressed overwhelmed"),
    ("sobrang pagod",         "very tired exhausted"),
    ("grabe naman",           "overwhelming too much frustrating"),
    ("grabe talaga",          "really overwhelming very much"),
    ("grabe",                 "overwhelming intense too much"),
    ("charot lang",           "just joking not serious"),
    ("charot",                "just joking not serious"),
    ("char",                  "just joking"),
    ("petmalu",               "amazing excellent great"),
    ("lodi",                  "idol amazing great person"),
    ("werpa",                 "great energy excellent"),
    ("sana ol",               "wish everyone had this lucky happy"),
    ("sana all",              "wish everyone had this lucky happy"),
    ("edi wow",               "sarcastic unimpressed eye roll"),   # sarcasm
    ("ano ba",                "frustrated confused what"),
    ("ay nako",               "frustrated overwhelmed oh no"),
    ("nako",                  "oh no worried"),
    ("naman",                 "come on pleading frustrated"),
    ("keri",                  "okay fine can handle it"),
    ("kaya ko",               "I can do it confident motivated"),
    ("kaya natin",            "we can do it motivated together"),
    ("lutang",                "lost confused distracted"),
    ("lugmok",                "exhausted defeated drained"),
    ("praning",               "paranoid anxious nervous"),
    ("kinakabahan",           "nervous anxious scared"),
    ("nakakainis",            "annoying frustrating"),
    ("nakakapagod",           "exhausting tiring"),
    ("nakakastress",          "stressful overwhelming"),
    ("nakakaloka",            "overwhelming shocking unbelievable"),
    ("loka",                  "crazy overwhelming unbelievable"),
    ("loko",                  "crazy unbelievable"),
    ("tapos na",              "done finished completed"),
    ("ubos na",               "drained exhausted used up"),
    ("pagod na",              "tired exhausted"),
    ("malungkot",             "sad unhappy"),
    ("masaya",                "happy joyful"),
    ("iyak",                  "crying sad"),
    ("takot",                 "scared fearful anxious"),
    ("baliw",                 "crazy stressed overwhelmed"),
    ("naiinis",               "annoyed frustrated"),
    ("nag-iisa",              "alone lonely"),
    ("mag-isa",               "alone lonely isolated"),
    ("walang gana",           "unmotivated no energy drained"),
    ("wala sa mood",          "not in the mood low energy"),
    ("todo na",               "going all out giving everything"),
    ("go lang",               "keep going motivated"),
    ("laban",                 "fight keep going motivated"),
    ("kumpiyansa",            "confident trust"),
    ("nainis",                "annoyed frustrated"),
    ("naiiyak",               "about to cry sad emotional"),

    # Gen Z / internet slangs
    ("no cap",                "honestly truly not joking"),
    ("fr fr",                 "for real seriously"),
    ("fr",                    "for real seriously"),
    ("ngl",                   "not going to lie honestly"),
    ("istg",                  "I swear seriously"),
    ("ong",                   "on god seriously honestly"),
    ("lowkey",                "kind of secretly a little"),
    ("highkey",               "very much really openly"),
    ("its giving",            "it feels like it seems"),
    ("giving",                "it seems feeling like"),
    ("slay",                  "excellent amazing did great"),
    ("slayed",                "did excellent amazing performed great"),
    ("ate that",              "did excellent performed great"),
    ("main character",        "feeling important confident great"),
    ("understood the assignment", "did great excellent nailed it"),
    ("rent free",             "cannot stop thinking obsessed stuck"),
    ("living in my head",     "cannot stop thinking obsessed worried"),
    ("in my feels",           "emotional sad feeling a lot"),
    ("down bad",              "very sad desperate struggling"),
    ("touch grass",           "need a break rest outside"),
    ("i'm dead",              "very funny exhausted done"),
    ("im dead",               "very funny exhausted done"),
    ("dead",                  "exhausted done finished"),
    ("it's giving chaos",     "very chaotic overwhelming stressed"),
    ("not okay",              "struggling not fine stressed"),
    ("not ok",                "struggling not fine stressed"),
    ("im not ok",             "struggling very stressed not fine"),
    ("i'm not ok",            "struggling very stressed not fine"),
    ("i give up",             "hopeless giving up defeated"),
    ("i cant",                "overwhelmed cannot handle frustrated"),
    ("i can't even",          "overwhelmed too much frustrated"),
    ("idk anymore",           "lost confused uncertain hopeless"),
    ("idk",                   "unsure not sure confused"),
    ("idek",                  "very confused lost uncertain"),
    ("crying rn",             "sad upset crying right now"),
    ("crying",                "sad upset emotional"),
    ("bestie",                "friend close person"),
    ("periodt",               "seriously that is the truth"),
    ("period",                "seriously final truth"),
    ("ate",                   "older sister close friend"),
    ("kuya",                  "older brother close friend"),
    ("burnout",               "burnout exhausted overwhelmed drained"),
    ("burnt out",             "burnout exhausted overwhelmed drained"),
    ("brain rot",             "mentally exhausted unfocused"),
    ("doom scrolling",        "distracted procrastinating avoidance"),
    ("ghosted",               "ignored abandoned left out"),
    ("toxic",                 "harmful stressful bad situation"),
    ("red flag",              "warning concern problem"),
    ("vibe check",            "checking how feeling mood"),
    ("vibing",                "relaxed content okay good"),
    ("rizz",                  "confident charming great"),
    ("mid",                   "mediocre average okay nothing special"),
    ("it is what it is",      "accepting resigned making peace"),
    ("at this point",         "exhausted resigned frustrated"),
    ("the audacity",          "shocked frustrated disbelief"),
    ("ate it",                "did great nailed it"),
    ("sending me",            "very funny laughing"),
    ("im so done",            "exhausted giving up frustrated"),
    ("i'm so done",           "exhausted giving up frustrated"),
    ("help",                  "struggling need support overwhelmed"),
    ("pls",                   "please stressed urgency"),
    ("plss",                  "please very stressed urgency"),
    ("fml",                   "frustrated terrible bad situation"),
    ("smh",                   "disappointed frustrated shaking head"),
    ("tbh",                   "honestly truthfully"),
    ("rn",                    "right now currently"),
    ("imo",                   "in my opinion I think"),
    ("irl",                   "in real life actually"),
    ("bruh",                  "frustrated unbelievable disappointed"),
    ("bruhh",                 "very frustrated unbelievable"),
    ("bro",                   "casual expression friend"),
    ("bro what",              "confused shocked disbelief"),
    ("sus",                   "suspicious uncertain uncomfortable"),
    ("based",                 "honest real authentic"),
    ("yikes",                 "uncomfortable cringe concerned"),
    ("big yikes",             "very uncomfortable very concerned"),
    ("mood",                  "relatable feeling same way"),
    ("same",                  "relatable agree feeling same"),
    ("ok boomer",             "dismissive frustrated eye roll"),    # sarcasm
    ("sure jan",              "sarcastic disbelief eye roll"),      # sarcasm
    ("sure",                  "okay fine accepting"),
    ("totally",               "completely absolutely"),
    ("whatever",              "indifferent resigned dismissive"),
]

# ── 4. Sarcasm detection patterns ───────────────────────────────────────────
# Each entry: (regex_pattern, sarcasm_type)
# Types: 'eye_roll', 'bitter', 'self_deprecating', 'forced_positivity'
SARCASM_PATTERNS = [
    # Forced positivity / irony
    (r"\b(oh\s+great|oh\s+wonderful|oh\s+fantastic|oh\s+perfect)\b",          "forced_positivity"),
    (r"\b(great|amazing|wonderful|fantastic|perfect)\b.{0,20}(not|never)",     "forced_positivity"),
    (r"totally\s+fine\b",                                                       "forced_positivity"),
    (r"\b(love|loving)\s+(that|this)\s+for\s+(me|us)\b",                       "forced_positivity"),
    (r"this\s+is\s+fine\b",                                                     "forced_positivity"),
    (r"\b(just|totally|absolutely)\s+(love|loving)\s+(failing|struggling|drowning|burning)", "forced_positivity"),
    (r"great\s+time\s+to\s+(fail|give up|cry|panic|breakdown)",                "forced_positivity"),
    (r"\b(sure|yep|yeah)\s*[,.]?\s*(totally|definitely|absolutely|of course)",  "forced_positivity"),
    (r"wow\s+(so|very|really|such)\s+(helpful|useful|great|amazing)",          "forced_positivity"),

    # Eye-roll / dismissive sarcasm
    (r"\b(edi\s+wow|edi\s+ikaw\s+na|sure\s+jan|ok\s+lang|ok\s+boomer)\b",     "eye_roll"),
    (r"\b(wow|amazing|incredible)\b.{0,15}(😒|🙄|💀|☠️)",                    "eye_roll"),
    (r"(🙄|😒).{0,30}(great|amazing|wonderful|fine|okay|sure)",               "eye_roll"),
    (r"\b(thanks\s+for\s+nothing|thanks\s+a\s+lot)\b",                         "eye_roll"),
    (r"\b(oh\s+sure|oh\s+yeah\s+right|right\s+right\s+right)\b",              "eye_roll"),
    (r"\b(obviously|clearly|of\s+course)\b.{0,20}(not|never|no)",             "eye_roll"),

    # Bitter sarcasm (negative situation dressed as positive)
    (r"\b(love|loving)\s+(all-nighter|cramming|failing|failing|crying)",       "bitter"),
    (r"(can't|cannot)\s+wait\s+to\s+(fail|cry|panic|breakdown|disappoint)",   "bitter"),
    (r"\b(so\s+fun|so\s+cool|so\s+great)\b.{0,20}(deadline|exam|fail|fail)",  "bitter"),
    (r"\b(best|greatest|most\s+amazing)\b.{0,20}(deadline|exam|groupmates|semester|grade)", "bitter"),
    (r"what\s+a\s+(great|wonderful|amazing|perfect)\s+(day|week|semester|time)\b", "bitter"),
    (r"\b(thriving|doing\s+great|doing\s+amazing)\b.{0,20}(not|barely|no\s+sleep|all-nighter)", "bitter"),
    (r"totally\s+(prepared|ready|fine|okay)\b.{0,20}(not|never|lol|haha|😂|💀)", "bitter"),

    # Self-deprecating sarcasm
    (r"\b(i'm\s+so\s+smart|so\s+smart\s+of\s+me)\b.{0,30}(forgot|missed|failed|didn't)", "self_deprecating"),
    (r"genius\s+(move|decision|idea)\b.{0,20}(forgot|missed|failed|stayed\s+up|lost)", "self_deprecating"),
    (r"\b(clearly|obviously)\s+a\s+genius\b",                                  "self_deprecating"),
    (r"\b(nailed\s+it|crushed\s+it)\b.{0,20}(not|barely|lol|just\s+kidding|💀|😂)", "self_deprecating"),
    (r"proud\s+of\s+myself\s+for\s+(forgetting|failing|missing|procrastinating)", "self_deprecating"),
]

# ── Sarcasm response bank ────────────────────────────────────────────────────
SARCASM_RESPONSES = {
    "forced_positivity": [
        "I notice you might be putting a brave face on things — and that's okay. How are you really doing underneath all that?",
        "It sounds like things might not actually be fine, even if you're saying they are. It's safe to be honest here — what's really going on?",
        "Sometimes we say 'it's fine' when it really isn't. You don't have to perform okayness here. What are you actually feeling?",
        "That sounds a little like you're being hard on yourself with a smile. What's really been weighing on you?",
    ],
    "eye_roll": [
        "I can sense some frustration in that! Totally valid. What's been getting to you lately?",
        "Ha — sounds like something (or someone) has been really testing your patience. Want to talk about it?",
        "I hear that eye-roll energy. Something's clearly not sitting right. What happened?",
        "Sounds like you're not exactly thrilled right now. What's been going on?",
    ],
    "bitter": [
        "I can feel the exhaustion in what you wrote — and I get it. School can be relentless. What's been the hardest part lately?",
        "That sounds less like excitement and more like you're running on empty. How long have things felt this rough?",
        "It sounds like you might be going through it right now, even if you're making light of it. What's really happening?",
        "I can hear how drained you are underneath that. You don't have to make it sound okay — what's actually going on?",
    ],
    "self_deprecating": [
        "I notice you're being a bit hard on yourself there. Mistakes happen to everyone, especially with how much pressure students carry. How are you really feeling about it?",
        "That self-deprecating humor tells me something might be bothering you more than you're letting on. What's going on?",
        "Being tough on yourself is easy when you're stressed. But you deserve some grace too. What's been happening?",
        "It's okay not to have it all together. What's been weighing on you lately?",
    ],
}


# ── Public API ───────────────────────────────────────────────────────────────

def normalize(text: str) -> tuple[str, bool, str | None]:
    """
    Normalize raw user text for the sentiment model.

    Returns:
        enriched_text (str)   — text safe to feed into TF-IDF
        is_sarcasm (bool)     — True if sarcasm was detected
        sarcasm_type (str|None) — one of forced_positivity / eye_roll /
                                   bitter / self_deprecating, or None
    """
    result = text

    # Step 1: emoji → words (operate on original before lowercasing
    #         so we catch the actual Unicode codepoints)
    for emoji_char, words in EMOJI_MAP.items():
        result = result.replace(emoji_char, f" {words} ")

    # Step 2: lowercase for everything below
    result = result.lower()

    # Step 3: text emoticons → words
    for pattern, replacement in EMOTICON_MAP:
        result = re.sub(pattern, f" {replacement} ", result, flags=re.IGNORECASE)

    # Step 4: slangs → plain English (longest-first order in SLANG_MAP)
    for slang, replacement in SLANG_MAP:
        # word-boundary-aware replace
        escaped = re.escape(slang)
        result = re.sub(rf"\b{escaped}\b", f" {replacement} ", result, flags=re.IGNORECASE)

    # Step 5: sarcasm detection — run on NORMALIZED text (emojis + slangs expanded)
    is_sarcasm = False
    sarcasm_type = None
    for pattern, s_type in SARCASM_PATTERNS:
        if re.search(pattern, result, flags=re.IGNORECASE):
            is_sarcasm = True
            sarcasm_type = s_type
            break

    # Clean up extra whitespace
    result = re.sub(r"\s+", " ", result).strip()

    return result, is_sarcasm, sarcasm_type


def get_sarcasm_response(sarcasm_type: str) -> str:
    import random
    bank = SARCASM_RESPONSES.get(sarcasm_type, SARCASM_RESPONSES["bitter"])
    return random.choice(bank)