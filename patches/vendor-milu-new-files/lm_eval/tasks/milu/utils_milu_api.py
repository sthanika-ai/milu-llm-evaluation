SYSTEM_INSTRUCTION = (
    "You are answering a multiple-choice question. Read the question and the four "
    'options, then respond with ONLY a JSON object of the form {"answer": "A"} '
    '(or "B", "C", "D") -- no other text.'
)


def doc_to_text_api(doc) -> str:
    choices = [doc["option1"], doc["option2"], doc["option3"], doc["option4"]]
    option_choices = {"A": choices[0], "B": choices[1], "C": choices[2], "D": choices[3]}

    prompt = SYSTEM_INSTRUCTION + "\n\nQuestion: " + doc["question"] + "\nChoices:\n"
    for choice, option in option_choices.items():
        prompt += f"{choice.upper()}. {option}\n"

    return prompt


def doc_to_target_letter(doc) -> str:
    target = doc["target"]
    option_number = ["1", "2", "3", "4"].index(target.split("option")[1])
    return "ABCD"[option_number]
