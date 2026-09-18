"""AI Orchestrator: intent -> plan -> Gemini call -> tool loop -> response."""
from __future__ import annotations

import base64
import os

from google import genai
from google.genai import types

from .memory import memory_manager
from .tools import TOOL_DECLARATIONS, run_tool

MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")

SYSTEM_INSTRUCTION = (
    "You are a helpful, precise multimodal AI assistant. "
    "Use tools when they let you give a more accurate answer. "
    "Keep responses concise and written in plain natural language sentences. "
    "Do not use markdown formatting such as **bold**, *italics*, bullet points, or headers."
)


def _client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")
    return genai.Client(api_key=api_key)


async def handle_message(
    session_id: str, user_text: str, attachments: list[dict] | None = None
) -> str:
    """Core loop: log message, call Gemini (with tool use), return final text."""
    mem = memory_manager.get(session_id)
    attachment_note = (
        f"\n[Attached: {', '.join(a['name'] for a in attachments)}]" if attachments else ""
    )
    mem.add("user", f"{user_text}{attachment_note}".strip())

    client = _client()
    tool = types.Tool(function_declarations=TOOL_DECLARATIONS)
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        tools=[tool],
        max_output_tokens=512,
    )

    contents = mem.history_for_model()

    # attach the actual file bytes (image/video/doc) to this turn so Gemini can see them
    if attachments:
        last = contents[-1]
        for att in attachments:
            try:
                raw = base64.b64decode(att["data"])
            except Exception:  # noqa: BLE001
                continue
            last["parts"].append(
                {"inline_data": {"mime_type": att["mime_type"], "data": raw}}
            )

    # tool-use loop: allow up to a few round trips
    for _ in range(4):
        response = client.models.generate_content(
            model=MODEL, contents=contents, config=config
        )
        candidate = response.candidates[0]
        parts = candidate.content.parts

        function_calls = [p.function_call for p in parts if p.function_call]
        if not function_calls:
            final_text = response.text or ""
            mem.add("assistant", final_text)
            return final_text

        # execute tool calls, feed results back
        contents.append(candidate.content)
        for fc in function_calls:
            result = run_tool(fc.name, dict(fc.args))
            contents.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_function_response(
                            name=fc.name, response={"result": result}
                        )
                    ],
                )
            )

    final_text = "I ran out of tool-call turns trying to answer that."
    mem.add("assistant", final_text)
    return final_text
