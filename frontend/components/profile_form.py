"""Household profile intake form rendered in the Streamlit sidebar.

Every widget has a stable session_state key so the voice input feature
(components/voice_input.py) can pre-fill values extracted from a spoken
transcript before the form re-renders.
"""
from __future__ import annotations

import streamlit as st

OCCUPATIONS = [
    "Farmer / Agricultural worker",
    "Daily wage / Unorganized sector worker",
    "Self-employed / Small business owner",
    "Salaried employee",
    "Homemaker",
    "Student",
    "Unemployed",
    "Retired",
    "Other",
]

INDIAN_STATES = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa", "Gujarat",
    "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh",
    "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab",
    "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh",
    "Uttarakhand", "West Bengal", "Delhi (NCT)", "Jammu and Kashmir", "Ladakh",
]

CATEGORIES = ["General", "SC", "ST", "OBC", "Minority"]

# Widget session_state keys, shared with voice_input.py's pre-fill logic.
FIELD_KEYS = {
    "age": "field_age",
    "gender": "field_gender",
    "annual_income": "field_annual_income",
    "family_size": "field_family_size",
    "occupation": "field_occupation",
    "state": "field_state",
    "category": "field_category",
    "is_student": "field_is_student",
    "is_disabled": "field_is_disabled",
    "is_farmer": "field_is_farmer",
    "is_bpl": "field_is_bpl",
    "free_text_context": "field_free_text_context",
}

DEFAULTS = {
    FIELD_KEYS["age"]: 35,
    FIELD_KEYS["gender"]: "female",
    FIELD_KEYS["annual_income"]: 100000,
    FIELD_KEYS["family_size"]: 4,
    FIELD_KEYS["occupation"]: OCCUPATIONS[0],
    FIELD_KEYS["state"]: INDIAN_STATES[0],
    FIELD_KEYS["category"]: CATEGORIES[0],
    FIELD_KEYS["is_student"]: False,
    FIELD_KEYS["is_disabled"]: False,
    FIELD_KEYS["is_farmer"]: False,
    FIELD_KEYS["is_bpl"]: False,
    FIELD_KEYS["free_text_context"]: "",
}


def ensure_defaults() -> None:
    for key, value in DEFAULTS.items():
        st.session_state.setdefault(key, value)


def render_profile_form() -> dict | None:
    ensure_defaults()

    with st.form("profile_form"):
        age = st.number_input("Age", min_value=0, max_value=120, key=FIELD_KEYS["age"])
        gender = st.selectbox(
            "Gender", ["female", "male", "other"], format_func=str.title, key=FIELD_KEYS["gender"]
        )
        annual_income = st.number_input(
            "Household annual income (INR)", min_value=0, step=5000, key=FIELD_KEYS["annual_income"]
        )
        family_size = st.number_input("Family size", min_value=1, key=FIELD_KEYS["family_size"])
        occupation = st.selectbox("Occupation", OCCUPATIONS, key=FIELD_KEYS["occupation"])
        state = st.selectbox("State / UT", INDIAN_STATES, key=FIELD_KEYS["state"])
        category = st.selectbox("Social category", CATEGORIES, key=FIELD_KEYS["category"])

        st.markdown("**Additional situations that apply**")
        is_student = st.checkbox("Student", key=FIELD_KEYS["is_student"])
        is_disabled = st.checkbox("Person with disability", key=FIELD_KEYS["is_disabled"])
        is_farmer = st.checkbox("Owns farmland", key=FIELD_KEYS["is_farmer"])
        is_bpl = st.checkbox("Below Poverty Line (BPL) household", key=FIELD_KEYS["is_bpl"])

        free_text_context = st.text_area(
            "Anything else, in your own words (optional)",
            placeholder="e.g. I recently lost my job, I am a widow raising two children...",
            key=FIELD_KEYS["free_text_context"],
        )

        submitted = st.form_submit_button("🔍 Find my schemes", use_container_width=True, type="primary")

    if not submitted:
        return None

    profile = {
        "age": int(age),
        "gender": gender,
        "annual_income": int(annual_income),
        "occupation": occupation,
        "state": state,
        "category": category,
        "is_student": is_student,
        "is_disabled": is_disabled,
        "is_farmer": is_farmer,
        "is_bpl": is_bpl,
        "family_size": int(family_size),
    }
    return {"profile": profile, "free_text_context": free_text_context or None}
