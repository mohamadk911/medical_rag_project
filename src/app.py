from retrieve import retrieve
from evidence_gate import check_evidence_gate
from generate_answer import generate_answer
from grounding_check import check_grounding, grounding_is_supported
from guardrails import validate_question, validate_retrieved_chunks, validate_answer_text
from llm_rerank import llm_rerank_chunks

FINAL_CONTEXT_K = 8


def main() -> None:
    print("Educational MDR RAG Assistant")
    print("Type 'exit' to stop.\n")

    while True:
        user_question = input("Ask MDR question: ")

        if user_question.lower().strip() in ["exit", "quit"]:
            break

        try:
            question = validate_question(user_question)

            print("\nRetrieving MDR chunks...")
            retrieved_chunks = retrieve(question, final_top_k=15)
            validate_retrieved_chunks(retrieved_chunks)

            print("\nReranking chunks with Gemini...")
            retrieved_chunks = llm_rerank_chunks(question, retrieved_chunks, top_k=FINAL_CONTEXT_K)
            validate_retrieved_chunks(retrieved_chunks)

            print("\nChecking MDR answerability...")
            gate_result = check_evidence_gate(question, retrieved_chunks)

            if not gate_result["passed"]:
                print("\n" + gate_result["message"])
                continue

            print("\nGenerating answer with Gemini...")
            answer = generate_answer(question, retrieved_chunks)

            if not validate_answer_text(answer):
                print("\nAnswer blocked because it may contain unsupported wording.")
                continue

            print("\nChecking grounding...")
            grounding_result = check_grounding(question, answer, retrieved_chunks)

            if grounding_is_supported(grounding_result):
                print("\nFinal answer:\n")
                print(answer)
            else:
                print("\nAnswer blocked by grounding checker.")
                print("\nGrounding result:\n")
                print(grounding_result)

        except Exception as error:
            print(f"\nError: {error}")

        print("\n" + "-" * 80 + "\n")


if __name__ == "__main__":
    main()