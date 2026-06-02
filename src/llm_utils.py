import time


def is_quota_error(error: Exception) -> bool:
    text = str(error).lower()
    return (
        "resource_exhausted" in text
        or "quota" in text
        or "429" in text
    )


def generate_content_with_retries(
    client,
    model: str,
    contents: str,
    max_retries: int = 3,
):
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            return client.models.generate_content(
                model=model,
                contents=contents,
            )
        except Exception as error:
            last_error = error

            if is_quota_error(error):
                raise

            if attempt == max_retries:
                raise

            time.sleep(attempt * 2)

    raise RuntimeError(f"LLM request failed: {last_error}")
