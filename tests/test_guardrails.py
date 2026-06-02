import sys
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from guardrails import validate_retrieved_chunks  # noqa: E402


def make_chunk(source_type: str = "MDR") -> dict:
    source = "MDR.pdf" if source_type == "MDR" else "mdcg_2021-24.pdf"

    return {
        "chunk_id": f"{source_type.lower()}_chunk_0001",
        "document_id": f"{source_type.lower()}_document",
        "source_type": source_type,
        "document_title": f"{source_type} document",
        "source": source,
        "text": "Regulatory source text.",
    }


class GuardrailsTest(unittest.TestCase):
    def test_mdr_chunks_pass_validation(self) -> None:
        validate_retrieved_chunks(
            [make_chunk("MDR")],
            expected_source_type="MDR",
        )

    def test_mdcg_chunks_pass_validation(self) -> None:
        validate_retrieved_chunks(
            [make_chunk("MDCG")],
            expected_source_type="MDCG",
        )

    def test_invalid_source_type_fails_validation(self) -> None:
        with self.assertRaisesRegex(ValueError, "Invalid source_type"):
            validate_retrieved_chunks([make_chunk("OTHER")])

    def test_empty_mdr_track_fails_validation(self) -> None:
        with self.assertRaisesRegex(ValueError, "No retrieved MDR chunks"):
            validate_retrieved_chunks([], expected_source_type="MDR")

    def test_empty_mdcg_track_can_be_allowed(self) -> None:
        validate_retrieved_chunks(
            [],
            expected_source_type="MDCG",
            allow_empty=True,
        )

    def test_prompt_no_longer_contains_rerank_marker_typo(self) -> None:
        prompt = (ROOT_DIR / "prompts" / "answer_prompt.txt").read_text(
            encoding="utf-8"
        )

        self.assertNotIn("definition directly.RETRIEVED", prompt)
        self.assertIn("cite the source document title instead", prompt)


if __name__ == "__main__":
    unittest.main()
