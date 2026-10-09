import os
from pathlib import Path

import httpx
import streamlit as st
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

API_URL = os.getenv("URL", "http://127.0.0.1:8000").rstrip("/")


def api_request(method: str, path: str, **kwargs) -> dict | list:
    try:
        response = httpx.request(
            method,
            f"{API_URL}{path}",
            timeout=180,
            **kwargs,
        )
        response.raise_for_status()
        return response.json()
    except httpx.ConnectError:
        st.error(
            "Cannot connect to the API. Start it in another terminal with "
            "`uvicorn app.main:app --reload`."
        )
        st.stop()
    except httpx.HTTPStatusError as error:
        try:
            detail = error.response.json().get("detail", error.response.text)
        except ValueError:
            detail = error.response.text
        st.error(f"API error ({error.response.status_code}): {detail}")
        st.stop()


st.set_page_config(page_title="Story Summarizer", page_icon="📚", layout="centered")
st.title("Story Summarizer")

sessions = api_request("GET", "/sessions")
if not sessions:
    created = api_request("POST", "/sessions", json={"title": "Story session"})
    st.session_state["active_session_id"] = created["id"]
    st.rerun()

if st.button("New session"):
    created = api_request("POST", "/sessions", json={"title": "Story session"})
    st.session_state["active_session_id"] = created["id"]
    st.rerun()

session_ids = [session["id"] for session in sessions]
active_session_id = st.session_state.get("active_session_id")
if active_session_id not in session_ids:
    active_session_id = session_ids[0]
    st.session_state["active_session_id"] = active_session_id

session_by_id = {session["id"]: session for session in sessions}
active_session_id = st.selectbox(
    "Session",
    options=session_ids,
    index=session_ids.index(active_session_id),
    format_func=lambda session_id: session_by_id[session_id]["title"],
    key="active_session_id",
)

session = api_request("GET", f"/sessions/{active_session_id}")
for summary in session["summaries"]:
    with st.expander(summary["story_name"]):
        st.markdown(summary["content"])

with st.form("story_form"):
    story_name = st.text_input("Story name", placeholder="Enter the story name here...")
    story = st.text_area(
        "Complete story",
        height=300,
        placeholder="Paste the full story here...",
    )
    submitted = st.form_submit_button("Summarize story", type="primary")

if submitted:
    if not story_name.strip() or not story.strip():
        st.error("Enter both a story name and the complete story.")
    else:
        with st.spinner("Reading and summarizing the story..."):
            result = api_request(
                "POST",
                f"/sessions/{active_session_id}/summaries",
                json={"story_name": story_name, "story": story},
            )
        st.markdown(result["content"])
