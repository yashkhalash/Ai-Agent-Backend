"""Minimal tool layer. Add real integrations (search, db, email...) as needed."""
from __future__ import annotations

import datetime


def get_current_time(_: dict) -> str:
    return datetime.datetime.now().isoformat()


def calculator(args: dict) -> str:
    expr = args.get("expression", "")
    try:
        # restricted eval: digits/operators only
        allowed = set("0123456789+-*/(). ")
        if not expr or not set(expr) <= allowed:
            return "error: invalid expression"
        return str(eval(expr, {"__builtins__": {}}, {}))
    except Exception as exc:  # noqa: BLE001
        return f"error: {exc}"


TOOL_REGISTRY = {
    "get_current_time": get_current_time,
    "calculator": calculator,
}

TOOL_DECLARATIONS = [
    {
        "name": "get_current_time",
        "description": "Get the current date and time.",
        "parameters": {"type": "OBJECT", "properties": {}},
    },
    {
        "name": "calculator",
        "description": "Evaluate a basic arithmetic expression.",
        "parameters": {
            "type": "OBJECT",
            "properties": {"expression": {"type": "STRING"}},
            "required": ["expression"],
        },
    },
]


def run_tool(name: str, args: dict) -> str:
    fn = TOOL_REGISTRY.get(name)
    if not fn:
        return f"error: unknown tool '{name}'"
    return fn(args)
