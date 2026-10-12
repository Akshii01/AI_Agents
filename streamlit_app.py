import os
from pathlib import Path

import httpx
import streamlit as st
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

API_URL = os.getenv("URL", "http://127.0.0.1:8000").rstrip("/")


def format_chat_content(content: str) -> str:
    return (
        content.replace("\\r\\n", "\n")
        .replace("\\n", "\n")
        .replace("\\r", "\n")
    )


def api_request(method: str, path: str, **kwargs) -> dict | list:
    require_auth = kwargs.pop("require_auth", True)
    headers = kwargs.pop("headers", {})
    token = st.session_state.get("access_token")
    if require_auth and token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        response = httpx.request(
            method,
            f"{API_URL}{path}",
            timeout=180,
            headers=headers,
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


st.set_page_config(page_title="AI Agent", page_icon="📚", layout="centered")

if not st.session_state.get("access_token"):
    st.title("AI Agent")
    login_tab, register_tab = st.tabs(["Log in", "Create account"])

    with login_tab:
        with st.form("login_form"):
            login_email = st.text_input("Email", key="login_email")
            login_password = st.text_input(
                "Password", type="password", key="login_password"
            )
            login_submitted = st.form_submit_button("Log in", type="primary")
        if login_submitted:
            result = api_request(
                "POST",
                "/auth/login",
                json={"email": login_email, "password": login_password},
                require_auth=False,
            )
            st.session_state["access_token"] = result["access_token"]
            st.session_state["user"] = result["user"]
            st.rerun()

    with register_tab:
        with st.form("register_form"):
            register_email = st.text_input("Email", key="register_email")
            register_password = st.text_input(
                "Password (at least 8 characters)",
                type="password",
                key="register_password",
            )
            register_submitted = st.form_submit_button(
                "Create account", type="primary"
            )
        if register_submitted:
            result = api_request(
                "POST",
                "/auth/register",
                json={"email": register_email, "password": register_password},
                require_auth=False,
            )
            st.session_state["access_token"] = result["access_token"]
            st.session_state["user"] = result["user"]
            st.rerun()
    st.stop()

current_user = api_request("GET", "/auth/me")
st.sidebar.caption(f"Signed in as {current_user['email']}")
if st.sidebar.button("Log out"):
    st.session_state.pop("access_token", None)
    st.session_state.pop("user", None)
    st.rerun()

agent_labels = {
    "story": "Story summarizer",
    "recipe": "Recipe chatbot",
    "general": "General chatbot",
}
agent_type = st.sidebar.radio(
    "Workspace",
    options=list(agent_labels),
    format_func=lambda value: agent_labels[value],
    key="active_agent_type",
)
st.title(agent_labels[agent_type])

sessions = api_request(
    "GET",
    "/sessions",
    params={"agent_type": agent_type},
)
session_ids = [session["id"] for session in sessions]
active_session_key = f"active_session_id_{agent_type}"
active_session_id = st.session_state.get(active_session_key)

if active_session_id not in session_ids:
    active_session_id = None

if st.sidebar.button("New session", key=f"new_session_{agent_type}"):
    default_titles = {
        "story": "Story session",
        "recipe": "Recipe chat",
        "general": "General chat",
    }
    created = api_request(
        "POST",
        "/sessions",
        json={
            "title": default_titles[agent_type],
            "agent_type": agent_type,
        },
    )
    st.session_state[active_session_key] = created["id"]
    st.rerun()

if not session_ids or active_session_id is None:
    default_titles = {
        "story": "Story session",
        "recipe": "Recipe chat",
        "general": "General chat",
    }
    created = api_request(
        "POST",
        "/sessions",
        json={
            "title": default_titles[agent_type],
            "agent_type": agent_type,
        },
    )
    st.session_state[active_session_key] = created["id"]
    st.rerun()

st.sidebar.subheader(f"{agent_labels[agent_type]} sessions")
for chat_session in sessions:
    session_id = chat_session["id"]
    session_title = chat_session["title"]
    if len(session_title) > 42:
        session_title = f"{session_title[:39].rstrip()}..."
    if st.sidebar.button(
        f"💬 {session_title}",
        key=f"select_session_{agent_type}_{session_id}",
        type="primary" if session_id == active_session_id else "secondary",
        use_container_width=True,
        help=chat_session["title"],
    ):
        st.session_state[active_session_key] = session_id
        st.rerun()

active_session_id = st.session_state[active_session_key]

session = api_request("GET", f"/sessions/{active_session_id}")

if agent_type == "story":
    for summary in session["summaries"]:
        with st.expander(summary["story_name"]):
            st.markdown(summary["content"])

    with st.form(f"story_form_{active_session_id}"):
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
                api_request(
                    "POST",
                    f"/sessions/{active_session_id}/summaries",
                    json={"story_name": story_name, "story": story},
                )
            st.rerun()

elif agent_type == "recipe":
    st.caption("Ask for recipes, ingredient substitutions, or cooking advice.")
    for message in session["recipe_chat_messages"]:
        with st.chat_message(message["role"]):
            st.markdown(format_chat_content(message["content"]))

    message = st.chat_input(
        "What would you like to cook?",
        key=f"recipe_chat_input_{active_session_id}",
    )
    if message:
        with st.spinner("Preparing a response..."):
            api_request(
                "POST",
                f"/sessions/{active_session_id}/recipe-chat",
                json={"message": message},
            )
        st.rerun()

else:
    st.caption(
        "Ask general questions. The assistant can look up current weather and search the web."
    )
    for message in session["general_chat_messages"]:
        with st.chat_message(message["role"]):
            st.markdown(format_chat_content(message["content"]))

    message = st.chat_input(
        "Ask me anything...",
        key=f"general_chat_input_{active_session_id}",
    )
    if message:
        with st.spinner("Thinking..."):
            api_request(
                "POST",
                f"/sessions/{active_session_id}/general-chat",
                json={"message": message},
            )
        st.rerun()
