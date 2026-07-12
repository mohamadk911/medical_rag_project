from __future__ import annotations

import hmac
import os
from pathlib import Path
import sys

from dotenv import load_dotenv
import streamlit as st


SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rag_core.pipeline import answer_question


load_dotenv()

st.set_page_config(
    page_title="MDR / MDCG Regulatory RAG Demo",
    page_icon="",
    layout="wide",
)


def check_password() -> bool:
    expected_password = os.getenv("STREAMLIT_APP_PASSWORD")

    if not expected_password:
        st.error("STREAMLIT_APP_PASSWORD is not configured in .env.")
        return False

    if st.session_state.get("password_ok"):
        return True

    with st.form("password_form"):
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Enter")

    if not submitted:
        return False

    if hmac.compare_digest(password, expected_password):
        st.session_state["password_ok"] = True
        st.rerun()

    st.error("Incorrect password.")
    return False


if not check_password():
    st.stop()


st.title("MDR / MDCG Regulatory RAG Demo")
st.warning(
    "This is a thesis prototype. Answers must be verified against the official MDR/MDCG sources."
)

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("top_k", min_value=4, max_value=18, value=18, step=1)
    enable_reranking = st.checkbox("Enable reranking", value=False)
    show_evidence = st.checkbox("Show evidence", value=True)
    show_metadata = st.checkbox("Show metadata", value=True)


question = st.text_area(
    "User question",
    height=140,
    placeholder="Example: What are the PRRC obligations for a medical device manufacturer under the MDR?",
)

ask_clicked = st.button("Ask", type="primary")


def source_label(source: dict) -> str:
    reference = source.get("reference")
    title = source.get("document_title") or source.get("title")
    source_type = source.get("source_type") or "Source"
    return f"{source_type}: {reference or title or source.get('document_id')}"


def evidence_label(item: dict, index: int) -> str:
    source_type = item.get("source_type") or "Source"
    reference = item.get("reference") or item.get("document_title") or item.get("document_id")
    pages = ""
    if item.get("page_start") is not None and item.get("page_end") is not None:
        pages = f", pages {item.get('page_start')}-{item.get('page_end')}"
    return f"{index}. {source_type} | {reference}{pages}"


if ask_clicked:
    if not question.strip():
        st.error("Please enter a question.")
    else:
        settings = {
            "top_k": top_k,
            "enable_reranking": enable_reranking,
            "enable_evidence_gate": False,
            "enable_grounding_check": False,
        }

        with st.spinner("Retrieving sources and generating answer..."):
            try:
                result = answer_question(question, settings=settings)
            except Exception as error:
                st.error(f"Error: {error}")
                st.stop()

        st.subheader("Final Answer")
        st.markdown(result.get("answer") or "No answer returned.")

        st.subheader("Sources Used")
        sources = result.get("sources", [])
        if sources:
            for source in sources:
                st.markdown(f"- {source_label(source)}")
        else:
            st.info("No sources returned.")

        if show_evidence:
            st.subheader("Evidence / Retrieved Chunks")
            evidence = result.get("evidence", [])
            if evidence:
                for index, item in enumerate(evidence, start=1):
                    with st.expander(evidence_label(item, index)):
                        st.json({
                            "chunk_id": item.get("chunk_id"),
                            "source_type": item.get("source_type"),
                            "document_id": item.get("document_id"),
                            "document_title": item.get("document_title"),
                            "reference": item.get("reference"),
                            "title": item.get("title"),
                            "page_start": item.get("page_start"),
                            "page_end": item.get("page_end"),
                            "hybrid_score": item.get("hybrid_score"),
                            "retrieval_source": item.get("retrieval_source"),
                        })
                        st.markdown(item.get("text") or "")
            else:
                st.info("No retrieved chunks returned.")

        if show_metadata:
            st.subheader("Metadata")
            st.json(result.get("metadata", {}))
