"""
Njiti website: a landing page with a question box and a chat page that streams the agent's answer.

Run from the repository root:
    uvicorn src.server:app --reload

The browser posts a question to /api/chat and reads back Server-Sent Events, one JSON object per event:
    {"type": "status", "status": "thinking"}          the agent has started
    {"type": "searching", "query": "..."}               the agent is looking something up
    {"type": "sources", "sources": [{title, url, kind}]} what it found
    {"type": "token", "text": "..."}                    a piece of the answer
    {"type": "error", "message": "..."}
    {"type": "done"}
"""

import asyncio
import json
import os
import re
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import BaseModel, Field

from src.premature_ai import WHO_FACT_SHEET, get_agent

WEB_DIR = Path(__file__).parent / "web"

# NJITI_DEMO=1 streams a canned answer so the site can be previewed without API keys
DEMO_MODE = os.getenv("NJITI_DEMO") == "1"

app = FastAPI(title="Njiti")
app.mount("/static", StaticFiles(directory=WEB_DIR / "static"), name="static")


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    thread_id: str = Field(..., min_length=8, max_length=64)


@app.get("/")
def home():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/chat")
def chat_page():
    return FileResponse(WEB_DIR / "chat.html")


@app.get("/api/health")
def health():
    return {"ok": True, "demo": DEMO_MODE}


@app.post("/api/chat")
async def chat(request: ChatRequest):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", request.thread_id):
        raise HTTPException(status_code=400, detail="Invalid thread_id")

    events = demo_events(request.message) if DEMO_MODE else agent_events(request.message, request.thread_id)
    return StreamingResponse(
        sse(events),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def sse(events):
    try:
        async for event in events:
            yield f"data: {json.dumps(event)}\n\n"
    except Exception as e:
        print(f"Chat error: {e!r}")
        yield f"data: {json.dumps({'type': 'error', 'message': 'Something went wrong while answering. Please try again.'})}\n\n"
    yield f"data: {json.dumps({'type': 'done'})}\n\n"


def message_text(message) -> str:
    """Text content of a message, whether the model returned a string or a list of content blocks"""
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text")


async def agent_events(message: str, thread_id: str):
    yield {"type": "status", "status": "thinking"}

    try:
        agent = get_agent()
    except RuntimeError as e:
        yield {"type": "error", "message": f"The assistant isn't set up yet: {e}."}
        return

    inputs = {"messages": [{"role": "user", "content": message}]}
    config = {"configurable": {"thread_id": thread_id}}

    async for mode, data in agent.astream(inputs, config=config, stream_mode=["messages", "updates"]):
        if mode == "messages":
            chunk, metadata = data
            # Chunks when the model streams; one whole message when it does not
            if metadata.get("langgraph_node") == "agent" and isinstance(chunk, AIMessage):
                text = message_text(chunk)
                if text:
                    yield {"type": "token", "text": text}

        elif mode == "updates":
            for node, update in data.items():
                for msg in (update or {}).get("messages", []):
                    if isinstance(msg, AIMessage):
                        for call in msg.tool_calls:
                            yield {"type": "searching", "query": call["args"].get("query", "")}
                    elif isinstance(msg, ToolMessage) and msg.artifact:
                        yield {"type": "sources", "sources": msg.artifact}


async def demo_events(message: str):
    yield {"type": "status", "status": "thinking"}
    await asyncio.sleep(0.8)
    yield {"type": "searching", "query": message}
    await asyncio.sleep(0.6)
    yield {"type": "sources", "sources": [
        WHO_FACT_SHEET,
        {"title": "WHO recommendations for care of the preterm or low-birth-weight infant (2022)", "url": None, "kind": "document"},
    ]}
    await asyncio.sleep(1.0)

    answer = (
        "This is a **demo answer** (the site is running with `NJITI_DEMO=1`, so no AI model was called).\n\n"
        "**Kangaroo mother care**\n\n"
        "- Hold the baby **skin-to-skin** on the chest, between the breasts, for as many hours a day as possible.\n"
        "- WHO recommends starting it **immediately after birth** for babies born too early, "
        "unless the baby is critically ill. [WHO](https://www.who.int/news-room/fact-sheets/detail/preterm-birth)\n"
        "- It keeps the baby warm, supports **breastfeeding**, and lowers the risk of death.\n\n"
        "**Take the baby to a health facility now** if they struggle to breathe, turn blue, stop feeding, or feel very cold."
    )
    for word in re.findall(r"\S+\s*", answer):
        yield {"type": "token", "text": word}
        await asyncio.sleep(0.03)
