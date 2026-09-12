import json
import hashlib
import hmac
import os
import re
import threading
import time
import traceback
from collections import OrderedDict, defaultdict, deque
from datetime import datetime

import gspread
from flask import Flask, Response, jsonify, render_template, request, session
from flask_cors import CORS
from google.oauth2.service_account import Credentials
from openai import OpenAI

from config import (
    CHATBOT_ID,
    ENABLE_GOOGLE_SHEETS,
    FLASK_SECRET_KEY,
    GOOGLE_SERVICE_ACCOUNT_ENV,
    LESSON_TYPE,
    MAX_HISTORY_MESSAGES,
    MAX_RESPONSE_TOKENS,
    MODEL_NAME,
    OPENAI_API_KEY_ENV,
    SPREADSHEET_ID,
    Stage,
    TEMPERATURE,
)
from data_loader import CHARACTERS
from dialogue_manager import get_food_answer, make_food_response
from food_utils import (
    clean_text,
    extract_like_object,
    find_food,
    is_safe_open_food_text,
    is_like_question,
    normalize_like_question,
    recognition_candidates,
    resolve_known_food,
)


app = Flask(__name__, static_folder=".", static_url_path="")
app.secret_key = FLASK_SECRET_KEY
CORS(app)

CHARACTER = CHARACTERS[CHATBOT_ID]
CHARACTER_NAME = CHARACTER["name"]
COUNTRY = CHARACTER["country"]
SHEET_TAB = CHARACTER.get("sheet_tab", CHATBOT_ID)
ENDING_MESSAGE = CHARACTER["ending_message"]

openai_key = os.environ.get(OPENAI_API_KEY_ENV)
openai_client = OpenAI(api_key=openai_key) if openai_key else None

tts_key = os.environ.get("OPENAI_TTS_API_KEY")
tts_client = OpenAI(api_key=tts_key, timeout=12.0, max_retries=0) if tts_key else None
TTS_MAX_CHARS = 500
TTS_RATE_LIMIT = 30
TTS_RATE_WINDOW_SECONDS = 60
TTS_CACHE_MAX_ITEMS = 256
tts_cache = OrderedDict()
tts_requests = defaultdict(deque)
stt_requests = defaultdict(deque)
tts_lock = threading.Lock()


sheet = None
if ENABLE_GOOGLE_SHEETS:
    try:
        raw_creds = os.environ.get(GOOGLE_SERVICE_ACCOUNT_ENV)
        if raw_creds:
            info = json.loads(raw_creds)
            creds = Credentials.from_service_account_info(
                info,
                scopes=[
                    "https://www.googleapis.com/auth/spreadsheets",
                    "https://www.googleapis.com/auth/drive",
                ],
            )
            spreadsheet = gspread.authorize(creds).open_by_key(SPREADSHEET_ID)
            try:
                sheet = spreadsheet.worksheet(SHEET_TAB)
            except gspread.exceptions.WorksheetNotFound:
                sheet = spreadsheet.add_worksheet(title=SHEET_TAB, rows=1000, cols=9)
            if not sheet.get_all_values():
                sheet.append_row([
                    "시간", "번호", "이름", "학생발화(보정)", "원본발화",
                    "루카응답", "단계", "나라", "수업유형",
                ])
            print(f"✅ 구글 시트 연결: {SHEET_TAB} / {LESSON_TYPE}")
        else:
            print("⚠️ GOOGLE_SERVICE_ACCOUNT 환경변수 없음")
    except Exception as error:
        print(f"❌ 구글 시트 연결 실패: {error}")
        traceback.print_exc()


def normalize_stage(stage):
    aliases = {
        "await_greeting": Stage.WAIT_GREETING.value,
        "WAIT_GREETING": Stage.WAIT_GREETING.value,
    }
    return aliases.get(stage, stage or Stage.WAIT_GREETING.value)


def select_recognition_candidate(primary, alternatives, stage):
    candidates = []
    for value in [primary, *(alternatives or [])]:
        candidate = str(value or "").strip()
        if candidate and candidate not in candidates:
            candidates.append(candidate)

    if not candidates:
        return ""

    if stage == Stage.WAIT_GREETING.value:
        return next((text for text in candidates if is_greeting(text)), candidates[0])

    if stage == Stage.WAIT_FEELING.value:
        return next(
            (
                text for text in candidates
                if normalize_feeling(text) and not is_feeling_question(text)
            ),
            candidates[0],
        )

    question_stage_values = {
        Stage.STUDENT_QUESTION_1.value,
        Stage.STUDENT_QUESTION_2.value,
        Stage.STUDENT_QUESTION_3.value,
    }
    if stage in question_stage_values:
        known_questions = [
            text for text in candidates
            if is_like_question(text) and find_food(text)
        ]
        if known_questions:
            return known_questions[0]
        return next(
            (text for text in candidates if is_like_question(text)),
            candidates[0],
        )

    return candidates[0]


def safe_login_value(value, fallback=""):
    value = str(value or "").strip()
    value = re.sub(r"[<>\r\n\t]", "", value)
    return value[:30] or fallback


def is_greeting(message):
    return normalize_greeting(message) is not None


def normalize_greeting(message):
    """Correct common Korean-accent greeting errors without changing login names."""
    text = clean_text(message)
    if not text:
        return None
    if "good morning" in text or "굿모닝" in text or "굿 모닝" in text:
        greeting = "Good morning"
    elif re.search(r"(?:^|\s)(?:hi|high)(?:\s|$)", text) or "하이" in text:
        greeting = "Hi"
    elif re.search(r"(?:^|\s)hey(?:\s|$)", text) or "헤이" in text:
        greeting = "Hey"
    elif (
        re.search(r"(?:^|\s)(?:hello|hallo|halo|yellow)(?:\s|$)", text)
        or "헬로" in text
        or "안녕" in text
    ):
        greeting = "Hello"
    else:
        normalized_name = clean_text(normalize_character_name(message))
        weak_hello = re.search(r"(?:^|\s)(?:call|low)(?:\s|$)", text)
        if not (weak_hello and clean_text(CHARACTER_NAME) in normalized_name):
            return None
        greeting = "Hello"
    return f"{greeting}, {CHARACTER_NAME}."


def normalize_character_name(message):
    text = str(message or "").strip()
    aliases = [CHARACTER_NAME, *CHARACTER.get("name_aliases", [])]
    aliases = sorted(
        {str(alias).strip() for alias in aliases if str(alias).strip()},
        key=len,
        reverse=True,
    )
    if not aliases:
        return text
    pattern = r"(?<!\w)(?:" + "|".join(re.escape(alias) for alias in aliases) + r")(?!\w)"
    return re.sub(pattern, CHARACTER_NAME, text, flags=re.IGNORECASE)


def parse_yes_no(message):
    text = clean_text(message)
    negatives = ["no", "no i dont", "no i don t", "i dont", "i don t", "do not", "아니"]
    positives = ["yes", "yes i do", "i do", "응", "네"]
    if any(value in text for value in negatives):
        return "no"
    if any(value in text for value in positives):
        return "yes"
    return None


FEELING_FORMS = (
    ("not bad", "I'm not bad.", "neutral"),
    ("so so", "I'm so-so.", "neutral"),
    ("not good", "I'm not good.", "negative"),
    ("unhappy", "I'm unhappy.", "negative"),
    ("wonderful", "I'm wonderful.", "positive"),
    ("fantastic", "I'm fantastic.", "positive"),
    ("excited", "I'm excited.", "positive"),
    ("awesome", "I'm awesome.", "positive"),
    ("perfect", "I'm perfect.", "positive"),
    ("nervous", "I'm nervous.", "negative"),
    ("scared", "I'm scared.", "negative"),
    ("sleepy", "I'm sleepy.", "negative"),
    ("hungry", "I'm hungry.", "negative"),
    ("bored", "I'm bored.", "negative"),
    ("angry", "I'm angry.", "negative"),
    ("tired", "I'm tired.", "negative"),
    ("upset", "I'm upset.", "negative"),
    ("sick", "I'm sick.", "negative"),
    ("sad", "I'm sad.", "negative"),
    ("happy", "I'm happy.", "positive"),
    ("great", "I'm great.", "positive"),
    ("good", "I'm good.", "positive"),
    ("fine", "I'm fine.", "positive"),
    ("okay", "I'm okay.", "neutral"),
    ("ok", "I'm okay.", "neutral"),
    ("cold", "I'm cold.", "negative"),
    ("hot", "I'm hot.", "negative"),
    ("bad", "I'm bad.", "negative"),
)


def is_feeling_question(message):
    return bool(re.search(r"\b(?:are you|how are you)\b", clean_text(message)))


def normalize_feeling(message):
    text = clean_text(message)
    if not text or is_feeling_question(text):
        return None
    text = re.sub(r"\bi m find\b|\bi am find\b", "i am fine", text)
    text = re.sub(r"\bi m tire\b|\bi am tire\b", "i am tired", text)
    text = re.sub(r"\bi m exciting\b|\bi am exciting\b", "i am excited", text)
    for phrase, display, category in FEELING_FORMS:
        if re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text):
            return {"display": display, "category": category}
    return None


def feeling_category(message):
    normalized = normalize_feeling(message)
    return normalized["category"] if normalized else "unknown"

def feeling_reply(message):
    category = feeling_category(message)
    if category == "negative":
        return "Oh, I see. Feel better soon. Let's travel together!"
    if category == "positive":
        return "Great! I'm happy, too. Now, let's travel together!"
    if category == "neutral":
        return "Okay! Let's travel together! Let's go!"
    return "That's okay! Let's travel together!"

def call_gpt(system_prompt, user_message, fallback):
    """제한된 생성형 피드백. 실패하면 수업 흐름을 지키는 기본 응답을 사용한다."""
    if not openai_client:
        return fallback
    try:
        result = openai_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_RESPONSE_TOKENS,
        )
        reply = (result.choices[0].message.content or "").strip()
        return reply or fallback
    except Exception as error:
        print(f"❌ OpenAI 호출 실패: {error}")
        return fallback


def classify_open_food_candidates(primary, alternatives):
    """Validate up to five open-vocabulary food candidates in one small AI call."""
    candidates = []
    for source_text in recognition_candidates(primary, alternatives):
        if not is_like_question(source_text):
            continue
        food_name = extract_like_object(source_text)
        if is_safe_open_food_text(food_name):
            candidates.append((source_text, food_name))
    if not candidates or not openai_client:
        return None

    prompt = """
You validate one food phrase from a Korean grade-3 English speaking activity.
Accept only when it clearly names ONE common, child-safe food, drink, fruit,
snack, ingredient, or dish. Reject unclear ASR fragments, people, brands,
body parts, abstract ideas, and medical, sexual, violent, or toilet words.
Return JSON only:
{"status":"accept" or "reject","candidate_index":0,"display_name":"natural English food name without a/an/the"}
candidate_index must identify the clearest acceptable phrase from the supplied list.
Do not guess or repair an unclear phrase.
""".strip()
    raw = call_gpt(
        prompt,
        json.dumps([name for _, name in candidates], ensure_ascii=False),
        '{"status":"reject","candidate_index":0,"display_name":""}',
    )
    try:
        result = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if result.get("status") != "accept":
        return None
    try:
        selected_index = int(result.get("candidate_index", 0))
        source_text = candidates[selected_index][0]
    except (TypeError, ValueError, IndexError):
        return None
    display_name = str(result.get("display_name") or "").strip().strip(".?!\"'")
    display_name = re.sub(r"^(?:a|an|the)\s+", "", display_name, flags=re.IGNORECASE)
    if not is_safe_open_food_text(display_name):
        return None
    cleaned_name = clean_text(display_name)
    return {
        "key": f"open:{cleaned_name}",
        "display_name": display_name,
        "source_text": source_text,
    }


def ai_feeling_reply(message):
    fallback = feeling_reply(message)
    prompt = f"""
You are {CHARACTER_NAME}, a friendly 10-year-old child from {COUNTRY}.
A Korean grade-3 beginner has answered "How are you today?"
Respond to the feeling in exactly one very short A1 English sentence.
Use no more than 5 words. Do not ask a question. Do not correct grammar.
Do not use emojis, explanations, Korean, or quotation marks.
""".strip()
    return call_gpt(prompt, message, fallback)


def ai_classify_yes_no(message):
    prompt = """
Classify a Korean grade-3 beginner's answer to a Do you like...? question.
Return exactly one lowercase word: yes, no, or unknown.
Accept short, imperfect, or mixed Korean-English answers.
""".strip()
    result = call_gpt(prompt, message, "unknown").lower().strip(" .!?\"'")
    return result if result in {"yes", "no"} else None


def save_log(corrected, original, reply, stage):
    if sheet is None:
        return
    try:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sheet.append_row([
            now,
            session.get("student_number", ""),
            session.get("student_name", ""),
            corrected,
            original,
            reply,
            stage,
            COUNTRY,
            LESSON_TYPE,
        ])
    except Exception as error:
        print(f"❌ 시트 저장 실패: {error}")
        traceback.print_exc()


def make_tts_token(text):
    message = str(text or "").strip().encode("utf-8")
    # 배포 환경의 TTS 키를 서명 비밀값으로 사용하므로 브라우저에서 임의의
    # 문장을 만들어 유료 TTS를 호출할 수 없다. 키 자체는 절대 전송하지 않는다.
    secret = str(tts_key or app.secret_key).encode("utf-8")
    return hmac.new(secret, message, hashlib.sha256).hexdigest()


def valid_tts_token(text, token):
    expected = make_tts_token(text)
    return bool(token) and hmac.compare_digest(expected, str(token))


def tts_rate_limited(client_id):
    now = time.monotonic()
    key = re.sub(r"[^A-Za-z0-9_-]", "", str(client_id or ""))[:80]
    if not key:
        key = (request.headers.get("X-Forwarded-For") or request.remote_addr or "unknown").split(",")[0]
    with tts_lock:
        recent = tts_requests[key]
        while recent and now - recent[0] > TTS_RATE_WINDOW_SECONDS:
            recent.popleft()
        if len(recent) >= TTS_RATE_LIMIT:
            return True
        recent.append(now)
    return False


def stt_rate_limited(client_id):
    now = time.monotonic()
    key = re.sub(r"[^A-Za-z0-9_-]", "", str(client_id or ""))[:80]
    if not key:
        key = (request.headers.get("X-Forwarded-For") or request.remote_addr or "unknown").split(",")[0]
    with tts_lock:
        recent = stt_requests[key]
        while recent and now - recent[0] > 60:
            recent.popleft()
        if len(recent) >= 20:
            return True
        recent.append(now)
    return False


def respond(
    reply,
    popup,
    next_stage,
    fireworks=False,
    original="",
    corrected=None,
    speech_reply=None,
    reaction="speaking",
    followup_reply=None,
):
    corrected = corrected if corrected is not None else original
    full_reply = " ".join(
        part.strip() for part in [reply, followup_reply] if part and part.strip()
    )
    history = session.get("chat_history", [])
    if corrected:
        history.append({"role": "user", "content": corrected})
    history.append({"role": "assistant", "content": full_reply})
    session["chat_history"] = history[-MAX_HISTORY_MESSAGES:]
    session.modified = True
    save_log(corrected, original, full_reply, next_stage)
    return jsonify({
        "reply": reply,
        "speech_reply": speech_reply or reply,
        "tts_token": make_tts_token(speech_reply or reply),
        "popup": popup,
        "stage": next_stage,
        "fireworks": fireworks,
        "recognized_text": corrected,
        "reaction": reaction,
        "followup_reply": followup_reply,
        "followup_tts_token": make_tts_token(followup_reply) if followup_reply else None,
    })


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/config", methods=["GET"])
def chatbot_config():
    images = CHARACTER.get("images", {})
    character_images = images.get("character", {})
    tts = CHARACTER.get("tts", {})
    return jsonify({
        "chatbotId": CHATBOT_ID,
        "characterName": CHARACTER_NAME,
        "country": COUNTRY,
        "landingTitle": CHARACTER.get("landing_title", "Hello, Korea!"),
        "gif": {
            "greeting": character_images.get("greeting", "greeting.gif"),
            "speaking": character_images.get("speaking", "speaking.gif"),
            "yes": character_images.get("yes", "yes.gif"),
            "no": character_images.get("no", "no.gif"),
        },
        "introBackground": (images.get("intro_backgrounds") or [images.get("intro_background", "")])[0],
        "introBackgrounds": images.get("intro_backgrounds", []),
        "backgrounds": images.get("backgrounds", []),
        "flagImg": images.get("flag", ""),
        "tts": {
            "provider": tts.get("provider", "openai"),
            "gender": tts.get("gender", CHARACTER.get("gender", "male")),
            "childMode": tts.get("child_mode", True),
            "rate": tts.get("rate", 0.85),
            "pitch": tts.get("pitch", 1.35),
        },
        "finaleMsg": CHARACTER.get("finale_message", "Come visit Italy next time!"),
        "homeUrl": CHARACTER.get("home_url", ""),
        # 로그인 화면이 보이는 동안 첫 인사 음성을 미리 준비할 수 있도록
        # 고정된 첫 문장과 해당 문장 전용 서명을 함께 보낸다.
        "introTtsText": CHARACTER["intro_speech"],
        "introTtsToken": make_tts_token(CHARACTER["intro_speech"]),
    })


@app.route("/api/tts", methods=["POST"])
def synthesize_speech():
    data = request.get_json(force=True, silent=True) or {}
    text = str(data.get("text") or "").strip()
    token = data.get("token")

    if not text or len(text) > TTS_MAX_CHARS:
        return jsonify({"error": "invalid_text"}), 400
    if not valid_tts_token(text, token):
        return jsonify({"error": "not_allowed"}), 403
    if tts_rate_limited(data.get("client_id")):
        return jsonify({"error": "rate_limited"}), 429
    if tts_client is None:
        return jsonify({"error": "tts_unavailable"}), 503

    tts = CHARACTER.get("tts", {})
    model = tts.get("model", "gpt-4o-mini-tts")
    voice = tts.get("voice", "cedar")
    instructions = tts.get(
        "instructions",
        "Speak clearly, warmly, and at a gentle pace for a young English learner.",
    )
    cache_key = hashlib.sha256(
        f"{model}\0{voice}\0{instructions}\0{text}".encode("utf-8")
    ).hexdigest()

    with tts_lock:
        cached = tts_cache.get(cache_key)
        if cached is not None:
            tts_cache.move_to_end(cache_key)
    if cached is not None:
        return Response(cached, mimetype="audio/mpeg", headers={"X-TTS-Cache": "HIT"})

    try:
        result = tts_client.audio.speech.create(
            model=model,
            voice=voice,
            input=text,
            instructions=instructions,
            response_format="mp3",
        )
        audio_bytes = result.content
        with tts_lock:
            tts_cache[cache_key] = audio_bytes
            tts_cache.move_to_end(cache_key)
            while len(tts_cache) > TTS_CACHE_MAX_ITEMS:
                tts_cache.popitem(last=False)
        return Response(audio_bytes, mimetype="audio/mpeg", headers={"X-TTS-Cache": "MISS"})
    except Exception as error:
        print(f"⚠️ OpenAI TTS 실패, 브라우저 음성으로 전환: {type(error).__name__}: {error}")
        return jsonify({"error": "tts_unavailable"}), 503


@app.route("/api/transcribe", methods=["POST"])
def transcribe_speech():
    """Apple 모바일에서는 불안정한 Safari 받아쓰기 대신 녹음 파일을 변환한다."""
    audio_file = request.files.get("audio")
    client_id = request.form.get("client_id")
    if audio_file is None:
        return jsonify({"error": "audio_required"}), 400
    if stt_rate_limited(client_id):
        return jsonify({"error": "rate_limited"}), 429
    if tts_client is None:
        return jsonify({"error": "stt_unavailable"}), 503

    audio_bytes = audio_file.read(4 * 1024 * 1024 + 1)
    if not audio_bytes or len(audio_bytes) > 4 * 1024 * 1024:
        return jsonify({"error": "invalid_audio"}), 400

    mime_type = audio_file.mimetype or "audio/mp4"
    filename = audio_file.filename or ("speech.webm" if "webm" in mime_type else "speech.m4a")
    prompt = (
        f"Transcribe only clearly audible English spoken by one Korean child to {CHARACTER_NAME}. "
        "The child may greet, say a feeling, answer yes or no, or ask Do you like plus one food. "
        "Never continue, complete, or invent speech during silence."
    )
    try:
        # Translation mode forces English output even when a Korean accent makes
        # the recognizer momentarily interpret a word as Korean.
        result = tts_client.audio.translations.create(
            model="whisper-1",
            file=(filename, audio_bytes, mime_type),
            prompt=prompt,
            temperature=0,
            response_format="json",
        )
        transcript = str(getattr(result, "text", "") or "").strip()
        if not transcript or re.search(r"[가-힣]", transcript):
            return jsonify({"error": "empty_transcript"}), 422
        normalized = clean_text(transcript)
        words = re.findall(r"[a-zA-Z']+", transcript)
        prompt_markers = (
            "the child may greet",
            "korean child",
            "transcribe only clearly audible english",
            "never continue complete or invent speech",
            "do you like plus one food",
        )
        feeling_examples = sum(
            phrase in normalized
            for phrase in ("i am happy", "i am good", "i am fine", "i am tired", "i am sad")
        )
        listed_foods = sum(
            re.search(rf"\b{re.escape(food)}\b", normalized) is not None
            for food in (
                "pizza", "pasta", "spaghetti", "hamburger", "taco", "ice cream",
                "sandwich", "sushi", "ramen", "rice", "bread", "cake", "cookie",
                "apple", "banana", "orange", "grape", "chicken", "fish", "egg",
                "cheese", "salad", "soup", "milk", "juice",
            )
        )
        if (
            len(words) > 14
            or any(marker in normalized for marker in prompt_markers)
            or feeling_examples >= 3
            or listed_foods >= 5
        ):
            print(f"⚠️ 비정상 STT 결과 차단: {transcript[:160]}")
            return jsonify({"error": "unreliable_transcript"}), 422
        return jsonify({"text": transcript})
    except Exception as error:
        print(f"⚠️ OpenAI STT 실패: {type(error).__name__}: {error}")
        return jsonify({"error": "stt_unavailable"}), 503

@app.route("/api/start", methods=["POST"])
def start_chat():
    data = request.get_json(force=True, silent=True) or {}
    student_number = safe_login_value(data.get("student_number"), "00")
    student_name = safe_login_value(data.get("student_name"), "친구")

    session.clear()
    session["student_number"] = student_number
    session["student_name"] = student_name
    session["asked_foods"] = []
    session["chat_history"] = []
    session["feeling_attempts"] = 0
    session["retry_mode"] = False
    session["question_retry_attempts"] = {}

    display_reply = f"Hi, {student_name}! {CHARACTER['intro_message']}"
    return respond(
        reply=display_reply,
        speech_reply=CHARACTER["intro_speech"],
        popup=f"{CHARACTER_NAME}에게 인사해 보세요.",
        next_stage=Stage.WAIT_GREETING.value,
    )


def question_retry_response(stage, original, ambiguity_options=None):
    """Give progressively stronger help without advancing the question stage."""
    attempts = session.get("question_retry_attempts", {})
    attempt = int(attempts.get(stage, 0)) + 1
    attempts[stage] = attempt
    session["question_retry_attempts"] = attempts
    session.modified = True

    if ambiguity_options and attempt < 3:
        first, second = ambiguity_options[:2]
        prompt = f"{first.capitalize()} or {second}? Please say it again."
        return respond(
            prompt,
            "다시 음식 이름을 또박또박 말해 보세요!",
            stage,
            original=original,
            corrected="",
        )

    if attempt == 1:
        return respond(
            'Try again! Please say, "Do you like ___?"',
            "음식 이름을 또박또박 말하며 다시 말해 보세요!",
            stage,
            original=original,
            corrected="",
            speech_reply='Try again! Please say, "Do you like?"',
        )

    if attempt == 2:
        return respond(
            'Say it slowly. "Do you... like... ___?"',
            "천천히 또박또박 다시 말해 보세요!",
            stage,
            original=original,
            corrected="",
            speech_reply='Say it slowly. "Do you like?"',
        )

    retry_examples = CHARACTER.get("retry_examples", {})
    default_examples = {
        Stage.STUDENT_QUESTION_1.value: "Do you like pizza?",
        Stage.STUDENT_QUESTION_2.value: "Do you like pasta?",
    }
    if stage in default_examples:
        example = str(retry_examples.get(stage, "")).strip()
        if not re.fullmatch(r"Do you like .+\?", example, flags=re.IGNORECASE):
            example = default_examples[stage]
        return respond(
            f'Let\'s try together. "{example}"',
            "화면의 문장을 천천히 따라 말해 보세요!",
            stage,
            original=original,
            corrected="",
            speech_reply=f"Let's try together. {example}",
        )

    return respond(
        "Choose an easy food. Try again!",
        "내가 발음하기 쉬운 음식으로 시도해보세요!",
        stage,
        original=original,
        corrected="",
    )


def clear_question_retry_attempts(stage):
    attempts = session.get("question_retry_attempts", {})
    if stage in attempts:
        attempts.pop(stage, None)
        session["question_retry_attempts"] = attempts
        session.modified = True


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True, silent=True) or {}
    stage = normalize_stage((data.get("stage") or "").strip())
    alternatives = data.get("alternatives")
    if not isinstance(alternatives, list):
        alternatives = []
    original = select_recognition_candidate(
        data.get("message"),
        alternatives[:5],
        stage,
    )

    if not original:
        return respond("Please say that again.", "다시 한 번 말해 보세요.", stage, original=original)

    # 로그인 화면의 한글 이름은 보존하되, 한글 STT 결과는 학생 말풍선에 띄우지 않는다.
    if stage != Stage.WAIT_GREETING.value and re.search(r"[가-힣]", original):
        return respond(
            "Please say that again in English.",
            "영어로 다시 한 번 말해 보세요.",
            stage,
            original=original,
            corrected="",
        )

    if stage == Stage.WAIT_GREETING.value:
        corrected_greeting = normalize_greeting(original)
        if not corrected_greeting:
            return respond(
                'Please say, "Hello!"',
                f'{CHARACTER_NAME}에게 "Hello!"라고 인사해 보세요.',
                stage,
                original=original,
            )
        return respond(
            f"Welcome to my country! This is {CHARACTER.get('country_question_name', COUNTRY)}.",
            "오늘의 기분을 영어로 말해 보세요.",
            Stage.WAIT_FEELING.value,
            original=original,
            corrected=corrected_greeting,
            followup_reply="How are you today?",
        )

    if stage == Stage.WAIT_FEELING.value:
        normalized_feeling = normalize_feeling(original)
        attempts = session.get("feeling_attempts", 0)
        if is_feeling_question(original) and attempts == 0:
            session["feeling_attempts"] = 1
            return respond(
                'Please say, "I\'m happy."',
                "오늘의 기분을 영어로 다시 말해 보세요.",
                Stage.WAIT_FEELING.value,
                original=original,
                corrected="",
            )
        if not normalized_feeling and attempts == 0:
            session["feeling_attempts"] = 1
            return respond(
                "How are you today?",
                "오늘의 기분을 영어로 다시 말해 보세요.",
                Stage.WAIT_FEELING.value,
                original=original,
                corrected="",
            )
        return respond(
            feeling_reply(original),
            "활동지를 보고 질문해 보세요.",
            Stage.STUDENT_QUESTION_1.value,
            original=original,
            corrected=normalized_feeling["display"] if normalized_feeling else "",
            followup_reply="Look! It's a food market!",
        )

    question_stages = {
        Stage.STUDENT_QUESTION_1.value: (
            Stage.STUDENT_QUESTION_2.value,
            "활동지를 보고 질문해 보세요.",
            "Ask me one more question.",
            0,
        ),
        Stage.STUDENT_QUESTION_2.value: (
            Stage.STUDENT_PREFERENCE.value,
            '“Yes, I do.” 또는 “No, I don’t.”로 대답해 보세요.',
            f"How about you? Do you like {CHARACTER.get('preference_food', 'ice cream')}?",
            1,
        ),
        Stage.STUDENT_QUESTION_3.value: (
            Stage.COUNTRY_PREFERENCE.value,
            '“Yes, I do.” 또는 “No, I don’t.”로 대답해 보세요.',
            f"Great! It was nice to see you here. Now, do you like {CHARACTER.get('country_question_name', COUNTRY)}?",
            2,
        ),
    }

    if stage in question_stages:
        food_resolution = resolve_known_food(data.get("message"), alternatives)
        if food_resolution["status"] == "ambiguous":
            return question_retry_response(
                stage,
                original,
                ambiguity_options=food_resolution.get("options"),
            )
        food = food_resolution.get("food")
        if food:
            original = food_resolution["source_text"]
        if not is_like_question(original):
            return question_retry_response(stage, original)

        if not food:
            food = classify_open_food_candidates(data.get("message"), alternatives)
            if food:
                original = food["source_text"]
        if not food:
            return question_retry_response(stage, original)

        corrected = f"Do you like {food['display_name']}?"
        clear_question_retry_attempts(stage)
        asked_foods = session.get("asked_foods", [])
        asked_key = food["key"]
        retry_mode = session.get("retry_mode", False)
        if asked_key in asked_foods and not retry_mode:
            return respond(
                "You already asked me that. Please choose a different food.",
                "다른 음식을 골라 질문해 보세요.",
                stage,
                original=original,
                corrected=corrected,
            )

        asked_foods.append(asked_key)
        session["asked_foods"] = asked_foods
        session["retry_mode"] = False
        next_stage, popup, followup_reply, question_number = question_stages[stage]
        answer = get_food_answer(COUNTRY, question_number)
        food_name = food["display_name"]
        reply = make_food_response(answer, food_name)
        return respond(
            reply,
            popup,
            next_stage,
            original=original,
            corrected=corrected,
            reaction=answer,
            followup_reply=followup_reply,
        )

    if stage == Stage.STUDENT_PREFERENCE.value:
        answer = parse_yes_no(original)
        if answer is None:
            return respond("Great try! Can you say that again?", '“Yes, I do.” 또는 “No, I don’t.”로 대답해 보세요.', stage, original=original)
        food_name = CHARACTER.get("preference_food", "ice cream")
        reply = f"Great! I like {food_name}, too." if answer == "yes" else "Okay! That's fine."
        corrected_answer = "Yes, I do." if answer == "yes" else "No, I don't."
        return respond(reply, "자유롭게 음식을 골라 질문해 보세요.", Stage.STUDENT_QUESTION_3.value, original=original, corrected=corrected_answer, reaction="yes", followup_reply="Good! Now, choose one more food and ask me.")

    if stage == Stage.COUNTRY_PREFERENCE.value:
        answer = parse_yes_no(original)
        if answer is None:
            return respond("Great try! Can you say that again?", '“Yes, I do.” 또는 “No, I don’t.”로 대답해 보세요.', stage, original=original)
        reply = ENDING_MESSAGE if answer == "yes" else "That's okay! I hope to see you again! Bye-bye!"
        corrected_answer = "Yes, I do." if answer == "yes" else "No, I don't."
        return respond(reply, None, Stage.END.value, fireworks=True, original=original, corrected=corrected_answer, reaction="yes")

    return respond(ENDING_MESSAGE, None, Stage.END.value, fireworks=True, original=original)


@app.route("/api/retry-question", methods=["POST"])
def retry_question():
    session["retry_mode"] = True
    attempts = session.get("question_retry_attempts", {})
    attempts.pop(Stage.STUDENT_QUESTION_3.value, None)
    session["question_retry_attempts"] = attempts
    session.modified = True
    return jsonify({
        "reply": "Ask me one more question.",
        "tts_token": make_tts_token("Ask me one more question."),
        "popup": "자유롭게 음식을 골라 다시 질문해 보세요.",
        "stage": Stage.STUDENT_QUESTION_3.value,
        "reaction": "speaking",
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
