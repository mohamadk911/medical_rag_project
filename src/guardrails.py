from typing import List, Dict


FORBIDDEN_PHRASES = [
    "according to my general knowledge",
    "based on external sources",
    "from the web",
    "outside the provided context",
]

ALLOWED_SOURCE_TYPES = {"MDR", "MDCG"}
REQUIRED_CHUNK_FIELDS = [
    "chunk_id",
    "document_id",
    "source_type",
    "document_title",
    "source",
    "text",
]


def validate_question(question: str) -> str:
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    return question.strip()


def validate_retrieved_chunks(
    chunks: List[Dict],
    expected_source_type: str | None = None,
    allow_empty: bool = False,
) -> None:
    if expected_source_type and expected_source_type not in ALLOWED_SOURCE_TYPES:
        raise ValueError(f"Unsupported expected source type: {expected_source_type}")

    if not chunks:
        if allow_empty:
            return
        label = expected_source_type or "regulatory"
        raise ValueError(f"No retrieved {label} chunks available.")

    for index, chunk in enumerate(chunks, start=1):
        source_type = chunk.get("source_type")

        if source_type not in ALLOWED_SOURCE_TYPES:
            raise ValueError(
                f"Invalid source_type for retrieved chunk {index}: {source_type}"
            )

        if expected_source_type and source_type != expected_source_type:
            raise ValueError(
                f"Expected {expected_source_type} chunk, got {source_type} "
                f"for chunk {chunk.get('chunk_id') or index}."
            )

        missing_fields = [
            field
            for field in REQUIRED_CHUNK_FIELDS
            if not chunk.get(field)
        ]

        if missing_fields:
            raise ValueError(
                f"Retrieved chunk {chunk.get('chunk_id') or index} is missing "
                f"required fields: {', '.join(missing_fields)}"
            )


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
