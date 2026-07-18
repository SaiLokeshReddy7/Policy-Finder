"""Renders the ranked scheme recommendations returned by /api/v1/navigate."""
from __future__ import annotations

import streamlit as st

from components.api_client import BackendError, synthesize_speech

VERDICT_ICONS = {
    "Likely Eligible": "🟢",
    "Possibly Eligible": "🟡",
    "Needs More Info": "🔵",
    "Not Eligible": "🔴",
}


def render_recommendations(response: dict) -> None:
    language = response.get("language", "en")

    for warning in response.get("warnings") or []:
        st.warning(warning)

    if response.get("profile_summary"):
        st.caption(f"Profile summary used for reasoning: {response['profile_summary']}")

    recommendations = response.get("recommendations", [])
    if not recommendations:
        st.info("No matching schemes were found for this profile.")
        return

    for rec in recommendations:
        icon = VERDICT_ICONS.get(rec["verdict"], "⚪")
        header = f"{icon} {rec['scheme_name']} — {rec['verdict']} (confidence {rec['confidence']:.0%})"
        with st.expander(header):
            st.markdown(f"**Category:** {rec['category']}")
            st.markdown(f"**Why:** {rec['reason']}")
            if rec.get("simplified_explanation"):
                st.markdown(f"**In plain language:** {rec['simplified_explanation']}")
                if st.button("🔊 Listen", key=f"listen_{rec['scheme_id']}"):
                    with st.spinner("Generating audio..."):
                        try:
                            audio_bytes, content_type = synthesize_speech(
                                rec["simplified_explanation"], language
                            )
                            st.audio(audio_bytes, format=content_type)
                        except BackendError as exc:
                            st.warning(str(exc))
            if rec.get("benefits"):
                st.markdown(f"**Benefits:** {rec['benefits']}")

            if rec.get("required_documents"):
                st.markdown("**Documents needed:**")
                for doc in rec["required_documents"]:
                    st.markdown(f"- {doc}")

            if rec.get("application_steps"):
                st.markdown("**How to apply:**")
                for i, step in enumerate(rec["application_steps"], start=1):
                    st.markdown(f"{i}. {step}")

            if rec.get("common_blockers"):
                st.markdown("**Common reasons applications get delayed:**")
                for blocker in rec["common_blockers"]:
                    st.markdown(f"- {blocker}")

            source = rec.get("source", {})
            origin_label = "Official knowledge base" if source.get("origin") == "knowledge_base" else "Live web search"
            st.markdown(f"**Source ({origin_label}):** [{source.get('name', 'link')}]({source.get('url', '#')})")
