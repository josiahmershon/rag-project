import os
from dotenv import load_dotenv
import chainlit as cl
import httpx

# Load environment variables from a local .env file if present
load_dotenv()

API_URL = os.getenv("BACKEND_API_URL", "http://localhost:8001")
DEFAULT_ENDPOINT = os.getenv("BACKEND_ENDPOINT", "/query-lc")
DEFAULT_USER_GROUPS = os.getenv("DEFAULT_USER_GROUPS", "executives")


def parse_user_groups(groups_str: str):
    return [g.strip() for g in groups_str.split(",") if g.strip()]


# --- Session helpers -------------------------------------------------

def set_groups(groups):
    cl.user_session.set("groups", groups)


def get_groups():
    return cl.user_session.get("groups", []) or []


# --- Chat lifecycle ---------------------------------------------------

@cl.on_chat_start
async def start():
    await cl.Message(
        content=(
            f"Connected to backend: {API_URL}{DEFAULT_ENDPOINT}\n\n"
            "Type your question."
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    query = (message.content or "").strip()
    if not query:
        await cl.Message(content="Please enter a question.").send()
        return

    user_groups = parse_user_groups(DEFAULT_USER_GROUPS)
    payload = {"query": query, "user_groups": user_groups}
    url = f"{API_URL}{DEFAULT_ENDPOINT}"

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        await cl.Message(content=f"Backend error: {e}").send()
        return

    answer = data.get("response", "(no response)")
    sources = data.get("sources", [])

    if sources:
        sources_text = "\n".join(
            f"- [{s.get('source','unknown')}]({s.get('source','unknown')})" for s in sources
        )
        content = f"{answer}\n\nSources:\n{sources_text}"
    else:
        content = answer

    await cl.Message(content=content).send()


