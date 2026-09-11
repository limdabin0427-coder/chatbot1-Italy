import re

from data_loader import FOODS


AMBIGUOUS_FOOD_PAIRS = {
    frozenset({"pizza", "pasta"}),
    frozenset({"rice", "french_fries"}),
    frozenset({"juice", "cheese"}),
    frozenset({"cake", "cookie"}),
    frozenset({"soup", "sushi"}),
}

SAFE_FOOD_CORRECTIONS = {
    "i scream": "ice_cream",
    "ice green": "ice_cream",
    "ice crim": "ice_cream",
    "posta": "pasta",
    "paster": "pasta",
    "spagetti": "spaghetti",
    "hamberger": "hamburger",
    "sandwitch": "sandwich",
    "ramyun": "ramen",
    "ramyeon": "ramen",
    "friedrise": "fried_rice",
    "kimchee": "kimchi",
    "bred": "bread",
    "cooky": "cookie",
    "chocorate": "chocolate",
    "grace": "grape",
    "solid": "salad",
    "jews": "juice",
    "choose": "juice",
    "macron": "macaron",
    "marathon": "malatang",
    "topokki": "tteokbokki",
    "kimbap": "gimbap",
}

BLOCKED_OPEN_WORDS = {
    "cancer", "sex", "sexy", "killing", "kill", "poop", "pee",
    "weapon", "gun", "knife", "drug", "drugs", "dead", "death",
}


def clean_text(text):
    text = str(text or "").lower().strip()
    text = re.sub(r"[^\w\s가-힣'-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _food_from_key(food_key):
    food_data = FOODS.get(food_key)
    if not food_data:
        return None
    return {
        "key": food_key,
        "display_name": food_data.get("display_name", food_key),
    }


def extract_like_object(text):
    cleaned = clean_text(text)
    match = re.search(r"\bdo\s+you\s+like\b\s+(.+)$", cleaned)
    if not match:
        return None
    object_name = match.group(1).strip()
    object_name = re.sub(r"^(?:a|an|the)\s+", "", object_name)
    return object_name or None


def is_like_question(text):
    return extract_like_object(text) is not None


def find_food(text):
    cleaned_text = clean_text(text)
    object_name = extract_like_object(text)
    correction_key = SAFE_FOOD_CORRECTIONS.get(clean_text(object_name))
    corrected_food = _food_from_key(correction_key)
    if corrected_food:
        return corrected_food

    food_items = sorted(
        FOODS.items(),
        key=lambda item: max(
            [len(alias) for alias in item[1].get("aliases", [])] + [len(item[0])]
        ),
        reverse=True,
    )
    for food_key, food_data in food_items:
        search_words = food_data.get("aliases", []) + [
            food_data.get("display_name", food_key)
        ]
        for word in sorted(set(search_words), key=len, reverse=True):
            cleaned_word = clean_text(word)
            if not cleaned_word:
                continue
            if re.search(rf"(?<!\w){re.escape(cleaned_word)}(?!\w)", cleaned_text):
                return _food_from_key(food_key)
    return None


def recognition_candidates(primary, alternatives=None):
    """Return at most five unique STT candidates without inventing speech."""
    candidates = []
    for value in [primary, *((alternatives or [])[:4])]:
        candidate = str(value or "").strip()
        if candidate and candidate not in candidates:
            candidates.append(candidate)
    return candidates


def resolve_known_food(primary, alternatives=None):
    """Prefer a registered food but pause on a genuinely confusable pair."""
    matches = []
    for candidate in recognition_candidates(primary, alternatives):
        if not is_like_question(candidate):
            continue
        food = find_food(candidate)
        if food:
            matches.append((candidate, food))

    matched_keys = {food["key"] for _, food in matches}
    ambiguous_pair = next(
        (pair for pair in AMBIGUOUS_FOOD_PAIRS if pair.issubset(matched_keys)),
        None,
    )
    if ambiguous_pair:
        options = [FOODS[key].get("display_name", key) for key in sorted(ambiguous_pair)]
        return {
            "status": "ambiguous",
            "source_text": str(primary or "").strip(),
            "options": options,
        }
    if matches:
        source_text, food = matches[0]
        return {"status": "known", "source_text": source_text, "food": food}
    return {"status": "unknown", "source_text": str(primary or "").strip()}


def is_safe_open_food_text(food_name):
    """Apply a strict local safety/shape gate before optional AI validation."""
    cleaned = clean_text(food_name)
    words = cleaned.split()
    if not cleaned or len(cleaned) > 50 or not 1 <= len(words) <= 5:
        return False
    if any(word in BLOCKED_OPEN_WORDS for word in words):
        return False
    return bool(re.fullmatch(r"[a-z0-9][a-z0-9 '\-]*", cleaned))


def normalize_like_question(text):
    food = find_food(text)
    name = food["display_name"] if food else extract_like_object(text)
    return f"Do you like {name}?" if name else str(text or "").strip()
