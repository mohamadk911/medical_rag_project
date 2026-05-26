from typing import List, Dict


FORBIDDEN_PHRASES = [
    "according to my general knowledge",
    "based on external sources",
    "from the web",
    "outside the provided context",
]


def validate_question(question: str) -> str:
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    return question.strip()


def validate_retrieved_chunks(chunks: List[Dict]) -> None:
    if not chunks:
        raise ValueError("No retrieved MDR chunks available.")

    for chunk in chunks:
        if chunk.get("source") != "MDR.pdf":
            raise ValueError("Invalid source detected. Only MDR.pdf chunks are allowed.")


def validate_answer_text(answer: str) -> bool:
    answer_lower = answer.lower()

    for phrase in FORBIDDEN_PHRASES:
        if phrase in answer_lower:
            return False

    return True


def educational_disclaimer() -> str:
    return (
        "Educational note: This answer is for learning purposes only and is not legal advice."
    )