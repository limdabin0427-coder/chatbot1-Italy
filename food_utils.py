import re

from data_loader import FOODS


def clean_text(text):
    """
    비교하기 쉽도록 문장을 정리합니다.
    """

    text = text.lower().strip()

    # 물음표, 마침표, 쉼표 같은 기호를 공백으로 바꿉니다.
    text = re.sub(r"[^\w\s가-힣]", " ", text)

    # 공백이 여러 개이면 하나로 줄입니다.
    text = re.sub(r"\s+", " ", text)

    return text


def find_food(text):
    """
    학생 문장에서 음식을 찾습니다.

    찾으면 아래 형식으로 반환합니다.

    {
        "key": "pizza",
        "display_name": "pizza"
    }

    찾지 못하면 None을 반환합니다.
    """

    cleaned_text = clean_text(text)

    # 긴 표현부터 먼저 확인합니다.
    # 예: "fried chicken"을 "chicken"보다 먼저 찾기 위함입니다.
    food_items = sorted(
        FOODS.items(),
        key=lambda item: max(
            len(alias) for alias in item[1]["aliases"]
        ),
        reverse=True
    )

    for food_key, food_data in food_items:
        aliases = food_data.get("aliases", [])

        # display_name도 검색 대상에 포함합니다.
        search_words = aliases + [
            food_data.get("display_name", food_key)
        ]

        # 긴 별칭부터 확인합니다.
        search_words = sorted(
            set(search_words),
            key=len,
            reverse=True
        )

        for word in search_words:
            cleaned_word = clean_text(word)

            if not cleaned_word:
                continue

            pattern = rf"(?<!\w){re.escape(cleaned_word)}(?!\w)"

            if re.search(pattern, cleaned_text):
                return {
                    "key": food_key,
                    "display_name": food_data.get(
                        "display_name",
                        food_key
                    )
                }

    return None


def is_like_question(text):
    """관사·단복수는 지적하지 않고 Do you like + 음식 문형만 확인합니다."""
    cleaned = clean_text(text)
    return bool(re.match(r"^do\s+you\s+like\s+.+$", cleaned))


def extract_like_object(text):
    cleaned = clean_text(text)
    match = re.match(r"^do\s+you\s+like\s+(.+)$", cleaned)
    return match.group(1).strip() if match else ""


def normalize_like_question(text):
    food = find_food(text)
    name = food["display_name"] if food else extract_like_object(text)
    return f"Do you like {name}?" if name else str(text or "").strip()
