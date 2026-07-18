"""Streamlit UI for the AI Citizen Scheme & Support Navigator.

Run with:
    streamlit run frontend/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from components.api_client import BackendError, chat, list_schemes, navigate  # noqa: E402
from components.profile_form import render_profile_form  # noqa: E402
from components.results_view import render_recommendations  # noqa: E402
from components.voice_input import render_voice_input  # noqa: E402

st.set_page_config(page_title="Citizen Scheme Navigator", page_icon="🧭", layout="wide")

st.title("AI Citizen Scheme & Support Navigator")
st.caption("Discover, verify, and understand the government welfare schemes you may be entitled to.")

st.sidebar.header("Your household profile")
with st.sidebar:
    render_voice_input()
    st.divider()
    form_result = render_profile_form()

if form_result is not None:
    with st.spinner("Finding schemes for you — checking eligibility, documents, and sources (about a minute)..."):
        try:
            response = navigate(form_result["profile"], form_result["free_text_context"])
            st.session_state["last_response"] = response
            st.session_state["chat_history"] = []  # new results -> fresh conversation
        except BackendError as exc:
            st.error(str(exc))

st.header("Recommended schemes for you")
if "last_response" in st.session_state:
    render_recommendations(st.session_state["last_response"])
else:
    st.info("👈 Fill in your profile (or speak it) and click **Find my schemes** to get started.")

with st.expander("📚 Browse all schemes in the knowledge base"):
    try:
        for scheme in list_schemes():
            st.markdown(f"**{scheme['name']}** ({scheme['category']}) — [{scheme['source']['name']}]({scheme['source']['url']})")
            st.caption(scheme["description"])
    except BackendError as exc:
        st.warning(f"Could not load the scheme list: {exc}")

# ---- Follow-up chat, grounded in the recommendations above ----
if "last_response" in st.session_state:
    st.divider()
    st.subheader("💬 Questions? Ask about your recommendations")
    st.caption("e.g. “Why am I not eligible for PM-KISAN?” · “Which scheme should I apply for first?” · “Where do I get an income certificate?”")

    history = st.session_state.setdefault("chat_history", [])
    for turn in history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])

    if question := st.chat_input("Type your question here..."):
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    resp = st.session_state["last_response"]
                    reply = chat(
                        message=question,
                        history=history,
                        profile_summary=resp.get("profile_summary", ""),
                        recommendations=resp.get("recommendations", []),
                    )
                except BackendError as exc:
                    reply = f"Sorry, I could not answer that: {exc}"
            st.markdown(reply)
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": reply})
