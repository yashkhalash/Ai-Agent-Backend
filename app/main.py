"""API Gateway + WebSocket chat endpoint."""
from __future__ import annotations

import os
import re
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from google.genai.errors import ClientError
from pydantic import BaseModel

from .orchestrator import handle_message

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"

load_dotenv(ENV_PATH)

app = FastAPI(title="AI Agent Gateway")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Attachment(BaseModel):
    name: str
    mime_type: str
    data: str  # base64-encoded file contents


class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str
    attachments: list[Attachment] | None = None


class ChatResponse(BaseModel):
    session_id: str
    reply: str


class ApiKeyStatus(BaseModel):
    configured: bool
    masked_key: str | None = None


class ApiKeyUpdate(BaseModel):
    api_key: str


def _mask(key: str) -> str:
    if len(key) <= 8:
        return "•" * len(key)
    return f"{key[:4]}{'•' * (len(key) - 8)}{key[-4:]}"


def _write_env_key(new_key: str | None) -> None:
    """Write or remove GEMINI_API_KEY in the .env file. None removes the line entirely."""
    pattern = re.compile(r"^GEMINI_API_KEY=")
    lines = ENV_PATH.read_text().splitlines() if ENV_PATH.exists() else []
    lines = [line for line in lines if not pattern.match(line)]
    if new_key:
        lines.append(f"GEMINI_API_KEY={new_key}")
    content = "\n".join(lines)
    ENV_PATH.write_text(f"{content}\n" if content else "")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/settings/api-key", response_model=ApiKeyStatus)
async def get_api_key_status() -> ApiKeyStatus:
    key = os.environ.get("GEMINI_API_KEY", "")
    return ApiKeyStatus(configured=bool(key), masked_key=_mask(key) if key else None)


@app.post("/settings/api-key", response_model=ApiKeyStatus)
async def set_api_key(req: ApiKeyUpdate) -> ApiKeyStatus:
    new_key = req.api_key.strip()
    if not new_key:
        raise HTTPException(400, "API key cannot be empty")

    # take effect immediately for this process, then persist so it survives a restart
    os.environ["GEMINI_API_KEY"] = new_key
    _write_env_key(new_key)

    return ApiKeyStatus(configured=True, masked_key=_mask(new_key))


@app.delete("/settings/api-key", response_model=ApiKeyStatus)
async def delete_api_key() -> ApiKeyStatus:
    os.environ.pop("GEMINI_API_KEY", None)
    _write_env_key(None)
    return ApiKeyStatus(configured=False, masked_key=None)


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    session_id = req.session_id or str(uuid.uuid4())
    attachments = [a.model_dump() for a in req.attachments] if req.attachments else None
    try:
        reply = await handle_message(session_id, req.message, attachments)
    except ClientError as exc:
        if exc.code == 429:
            raise HTTPException(429, "Gemini API quota exceeded. Add a new API key.") from exc
        raise HTTPException(exc.code or 500, str(exc)) from exc
    return ChatResponse(session_id=session_id, reply=reply)


@app.websocket("/ws/chat/{session_id}")
async def chat_ws(websocket: WebSocket, session_id: str) -> None:
    await websocket.accept()
    try:
        while True:
            user_text = await websocket.receive_text()
            reply = await handle_message(session_id, user_text)
            await websocket.send_text(reply)
    except WebSocketDisconnect:
        pass
