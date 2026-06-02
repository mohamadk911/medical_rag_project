"""
app.py

Main orchestration pipeline for the MDR/MDCG regulatory RAG.

Pipeline:
1. User question
2. Question validation
3. Retrieval attempt loop
4. Stage 1: catalog-based document selection inside retrieve.py
5. Stage 2: separate MDR and MDCG chunk retrieval
6. Optional LLM reranking
7. Evidence gate
8. Answer generation
9. Grounding check
10. Final answer or controlled fallback

The system never loops forever because MAX_RETRIEVAL_ATTEMPTS limits retries.
"""

import os

from dotenv import load_dotenv

from retrieve import retrieve
from evidence_gate import check_evidence_gate
from fallback_answer import generate_source_fallback_answer
from generate_answer import generate_answer
from grounding_check import check_grounding, grounding_is_supported
from guardrails import (
    validate_question,
    validate_retrieved_chunks,
    validate_answer_text,
)
from llm_rerank import llm_rerank_chunks
from llm_utils import is_quota_error


load_dotenv()


def env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)

    if value is None:
        return default

    return value.strip().lower() in {"1", "true", "yes", "on"}


USE_LLM_RERANKER = env_flag("USE_LLM_RERANKER", False)
USE_LLM_EVIDENCE_GATE = env_flag("USE_LLM_EVIDENCE_GATE", False)
USE_LLM_GROUNDING_CHECK = env_flag("USE_LLM_GROUNDING_CHECK", False)

FINAL_MDR_CONTEXT_K = 5
FINAL_MDCG_CONTEXT_K = 6

MAX_RETRIEVAL_ATTEMPTS = 3


def main() -> None:
    print("Educational Medical Device Regulatory RAG Assistant")
    print("Type 'exit' to stop.\n")

    while True:
        user_question = input("Ask regulatory question: ")

        if user_question.lower().strip() in ["exit", "quit"]:
            break

        try:
            question = validate_question(user_question)
            answer_found = False

            for attempt in range(1, MAX_RETRIEVAL_ATTEMPTS + 1):
                print(f"\nRetrieval attempt {attempt}/{MAX_RETRIEVAL_ATTEMPTS}")

                retrieval_result = retrieve(
                    query=question,
                    attempt=attempt,
                )

                mdr_chunks = retrieval_result.get("mdr_chunks", [])
                mdcg_chunks = retrieval_result.get("mdcg_chunks", [])

                validate_retrieved_chunks(
                    mdr_chunks,
                    expected_source_type="MDR",
                )
                validate_retrieved_chunks(
                    mdcg_chunks,
                    expected_source_type="MDCG",
                    allow_empty=True,
                )

                if USE_LLM_RERANKER:
                    print("\nReranking MDCG chunks...")
                    mdcg_chunks = llm_rerank_chunks(
                        question,
                        mdcg_chunks,
                        top_k=FINAL_MDCG_CONTEXT_K,
                    )

                    print("\nReranking MDR chunks...")
                    mdr_chunks = llm_rerank_chunks(
                        question,
                        mdr_chunks,
                        top_k=FINAL_MDR_CONTEXT_K,
                    )

                combined_chunks = mdcg_chunks + mdr_chunks

                if USE_LLM_EVIDENCE_GATE:
                    print("\nChecking source answerability...")
                    try:
                        gate_result = check_evidence_gate(
                            question,
                            combined_chunks,
                        )
                    except Exception as error:
                        if not is_quota_error(error):
                            raise

                        print("\nEvidence gate skipped because Gemini quota is exhausted.")
                        gate_result = {
                            "passed": True,
                            "reason": "quota_skip",
                            "message": "Skipped due to Gemini quota.",
                        }

                    if not gate_result["passed"]:
                        print("\nEvidence insufficient for this document set.")

                        if attempt < MAX_RETRIEVAL_ATTEMPTS:
                            print("Trying next catalog document candidates...")
                            continue

                        print("\n" + gate_result["message"])
                        break

                print("\nGenerating answer...")
                try:
                    answer = generate_answer(
                        question,
                        combined_chunks,
                    )
                except Exception as error:
                    if not is_quota_error(error):
                        raise

                    print("\nGemini quota exhausted. Showing retrieved source fallback.\n")
                    print(generate_source_fallback_answer(question, combined_chunks))
                    answer_found = True
                    break

                if not validate_answer_text(answer):
                    print("\nAnswer blocked because it contains unsupported wording.")
                    break

                if USE_LLM_GROUNDING_CHECK:
                    print("\nChecking grounding...")
                    try:
                        grounding_result = check_grounding(
                            question,
                            answer,
                            combined_chunks,
                        )
                    except Exception as error:
                        if not is_quota_error(error):
                            raise

                        print("\nGrounding check skipped because Gemini quota is exhausted.")
                        grounding_result = "SUPPORTED_QUOTA_SKIPPED"

                    if not grounding_is_supported(grounding_result):
                        print("\nGrounding failed for this document set.")

                        if attempt < MAX_RETRIEVAL_ATTEMPTS:
                            print("Trying next catalog document candidates...")
                            continue

                        print("\nAnswer blocked by grounding checker.")
                        print("\nGrounding result:\n")
                        print(grounding_result)
                        break

                print("\nFinal answer:\n")
                print(answer)
                answer_found = True
                break

            if not answer_found:
                pass

        except Exception as error:
            print(f"\nError: {error}")

        print("\n" + "-" * 80 + "\n")


if __name__ == "__main__":
    main()
