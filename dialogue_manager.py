from data_loader import ANSWER_PATTERNS


def get_answer_pattern(country):
    return ANSWER_PATTERNS.get(country, ["yes", "no", "yes"])


def get_food_answer(country, question_number):
    pattern = get_answer_pattern(country)
    if 0 <= question_number < len(pattern):
        return pattern[question_number]
    return pattern[-1] if pattern else "yes"


def make_food_response(answer, food_name, ask_back=False):
    if answer == "yes":
        reply = f"Yes, I do. I like {food_name}."
    else:
        reply = f"No, I don't. I don't like {food_name}."
    if ask_back:
        reply += f" Do you like {food_name}?"
    return reply


def compare_food_answers(bot_answer, student_answer, ending_question):
    if bot_answer == student_answer:
        return f"Oh, we are the same! That's great. {ending_question}"
    return f"That's okay. We can be different. {ending_question}"
