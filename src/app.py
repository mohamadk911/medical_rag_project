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

from retrieve import retrieve
from evidence_gate import check_evidence_gate
from generate_answer import generate_answer
from grounding_check import check_grounding, grounding_is_supported
from guardrails import (
    validate_question,
    validate_retrieved_chunks,
    validate_answer_text,
)
from llm_rerank import llm_rerank_chunks


USE_LLM_RERANKER = True

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

                validate_retrieved_chunks(mdr_chunks)
                validate_retrieved_chunks(mdcg_chunks)

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

                print("\nChecking source answerability...")
                gate_result = check_evidence_gate(
                    question,
                    combined_chunks,
                )

                if not gate_result["passed"]:
                    print("\nEvidence insufficient for this document set.")

                    if attempt < MAX_RETRIEVAL_ATTEMPTS:
                        print("Trying next catalog document candidates...")
                        continue

                    print("\n" + gate_result["message"])
                    break

                print("\nGenerating answer...")
                answer = generate_answer(
                    question,
                    combined_chunks,
                )

                if not validate_answer_text(answer):
                    print("\nAnswer blocked because it contains unsupported wording.")
                    break

                print("\nChecking grounding...")
                grounding_result = check_grounding(
                    question,
                    answer,
                    combined_chunks,
                )

                if grounding_is_supported(grounding_result):
                    print("\nFinal answer:\n")
                    print(answer)
                    answer_found = True
                    break

                print("\nGrounding failed for this document set.")

                if attempt < MAX_RETRIEVAL_ATTEMPTS:
                    print("Trying next catalog document candidates...")
                    continue

                print("\nAnswer blocked by grounding checker.")
                print("\nGrounding result:\n")
                print(grounding_result)

            if not answer_found:
                pass

        except Exception as error:
            print(f"\nError: {error}")

        print("\n" + "-" * 80 + "\n")


if __name__ == "__main__":
    main()