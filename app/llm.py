"""The only file that knows about LLM providers (Ollama and Amazon Bedrock).
Choose the provider with environment variables:
    LLM_PROVIDER=ollama   (default)
    LLM_PROVIDER=bedrock
"""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

from utils import pretty_print_json

PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
BEDROCK_REGION = os.getenv("BEDROCK_REGION", "eu-north-1")
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "amazon.nova-lite-v1:0")
MAX_TOKENS = 1000

logger = logging.getLogger("llm")


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict


@dataclass
class Reply:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int | None = None
    output_tokens: int | None = None
    raw: Any = None  # the provider's own response object, for debugging


def _clean_schema(s: Any) -> Any:
    """Remove the auto-generated 'title' strings Pydantic adds to JSON schemas."""
    if isinstance(s, dict):
        return {k: _clean_schema(v) for k, v in s.items() if not (k == "title" and isinstance(v, str))}
    if isinstance(s, list):
        return [_clean_schema(v) for v in s]
    return s


# ---------------------------------------------------------------- Ollama
def _to_ollama_messages(messages: list, system: str) -> list:
    out = [{"role": "system", "content": system}]
    for m in messages:
        if m["role"] == "assistant":
            d = {"role": "assistant", "content": m.get("content", "")}
            if m.get("tool_calls"):
                d["tool_calls"] = [{"function": {"name": c["name"], "arguments": c["args"]}} for c in m["tool_calls"]]
            out.append(d)
        elif m["role"] == "tool":
            out.append({"role": "tool", "tool_name": m["name"], "content": m["content"]})
        else:
            out.append({"role": "user", "content": m["content"]})
    return out


def _ollama_chat(messages: list, system: str, tools: list) -> Reply:
    import ollama

    extra = {"think": False} if "qwen3" in OLLAMA_MODEL else {}  # skip slow reasoning text
    resp = ollama.chat(
        model=OLLAMA_MODEL,
        messages=_to_ollama_messages(messages, system),
        tools=[
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": _clean_schema(t["parameters"]),
                },
            }
            for t in tools
        ],
        options={"temperature": 0},
        **extra,
    )
    msg = resp.message
    calls = [ToolCall(id=f"call_{i}", name=c.function.name, args=dict(c.function.arguments)) for i, c in enumerate(msg.tool_calls or [])]
    return Reply(
        text=msg.content or "",
        tool_calls=calls,
        input_tokens=resp.prompt_eval_count,
        output_tokens=resp.eval_count,
        raw=resp,
    )


def _ollama_structured(messages: list, system: str, schema: dict) -> str:
    import ollama

    extra = {"think": False} if "qwen3" in OLLAMA_MODEL else {}
    resp = ollama.chat(
        model=OLLAMA_MODEL,
        messages=_to_ollama_messages(messages, system),
        format=schema,
        options={"temperature": 0},
        **extra,
    )
    return resp.message.content or ""


# ---------------------------------------------------------------- Bedrock
_bedrock_client = None


def _client():
    global _bedrock_client
    if _bedrock_client is None:
        import boto3

        _bedrock_client = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)  # Create the one instace low level client
    return _bedrock_client


def _add(out: list, role: str, blocks: list) -> None:
    """Append a message, merging into the previous one if the role is the same
    (Bedrock requires user and assistant messages to alternate)."""
    if out and out[-1]["role"] == role:
        out[-1]["content"].extend(blocks)
    else:
        out.append({"role": role, "content": list(blocks)})


def _to_bedrock_messages(messages: list) -> list:
    out: list = []
    for m in messages:
        role = m["role"]
        if role == "tool":
            # All tool results for one assistant turn end up in a single user message.
            try:
                parsed = json.loads(m["content"])
            except json.JSONDecodeError:
                parsed = None
            content = [{"json": parsed}] if isinstance(parsed, dict) else [{"text": m["content"]}]
            _add(
                out,
                "user",
                [{"toolResult": {"toolUseId": m["tool_call_id"], "content": content}}],
            )
        elif role == "assistant":
            blocks = []
            if m.get("content"):
                blocks.append({"text": m["content"]})
            for c in m.get("tool_calls") or []:
                blocks.append(
                    {
                        "toolUse": {
                            "toolUseId": c["id"],
                            "name": c["name"],
                            "input": c["args"],
                        }
                    }
                )
            if not blocks:  # Bedrock rejects empty content
                blocks = [{"text": "(no output)"}]
            _add(out, "assistant", blocks)
        else:
            _add(out, "user", [{"text": m["content"]}])
    return out


def _bedrock_tool_config(tools: list, choice: dict | None = None) -> dict:
    cfg: dict = {
        "tools": [
            {
                "toolSpec": {
                    "name": t["name"],
                    "description": t["description"],
                    "inputSchema": {"json": _clean_schema(t["parameters"])},
                }
            }
            for t in tools
        ]
    }
    if choice:
        cfg["toolChoice"] = choice
    return cfg


def _bedrock_chat(messages: list, system: str, tools: list) -> Reply:
    resp = _client().converse(
        modelId=BEDROCK_MODEL_ID,
        system=[{"text": system}],
        messages=_to_bedrock_messages(messages),
        toolConfig=_bedrock_tool_config(tools),
        inferenceConfig={"temperature": 0, "maxTokens": MAX_TOKENS},
    )
    blocks = resp["output"]["message"]["content"]
    calls = [
        ToolCall(
            id=b["toolUse"]["toolUseId"],
            name=b["toolUse"]["name"],
            args=b["toolUse"]["input"],
        )
        for b in blocks
        if "toolUse" in b
    ]
    usage = resp.get("usage", {})
    return Reply(
        text="".join(b.get("text", "") for b in blocks),
        tool_calls=calls,
        input_tokens=usage.get("inputTokens"),
        output_tokens=usage.get("outputTokens"),
        raw=resp,
    )


def _bedrock_structured(messages: list, system: str, schema: dict, tools: list) -> str:
    """Converse has no 'response schema' option, so we force a 'final_answer' tool call."""
    final_tool = {
        "name": "final_answer",
        "description": "Return the final answer to the user.",
        "parameters": schema,
    }

    logger.info("Send structured message to bedrock, using final_answer tool")

    resp = _client().converse(
        modelId=BEDROCK_MODEL_ID,
        system=[{"text": system}],
        messages=_to_bedrock_messages(messages),
        # The history contains toolUse blocks, so the original tools stay in the config.
        toolConfig=_bedrock_tool_config(list(tools) + [final_tool], {"tool": {"name": "final_answer"}}),
        inferenceConfig={"temperature": 0, "maxTokens": MAX_TOKENS},
    )

    logger.debug("Structured response of bedrock is %s", pretty_print_json(resp))

    for b in resp["output"]["message"]["content"]:
        if "toolUse" in b and b["toolUse"]["name"] == "final_answer":
            logger.info("Response Received - Using final_answer tool")
            return json.dumps(b["toolUse"]["input"])

    logger.warning("No tool called return response as it is")
    return "".join(b.get("text", "") for b in resp["output"]["message"]["content"])


# ---------------------------------------------------------------- public API
def chat(messages: list, system: str, tools: list) -> Reply:
    """One model call with tools. Returns a provider-independent Reply."""
    if PROVIDER == "bedrock":
        return _bedrock_chat(messages, system, tools)
    return _ollama_chat(messages, system, tools)


def chat_structured(messages: list, system: str, schema: dict, tools: list) -> str:
    """A structured call that needs to match the JSON Schema"""
    if PROVIDER == "bedrock":
        return _bedrock_structured(messages, system, schema, tools)
    return _ollama_structured(messages, system, schema)


def describe() -> str:
    if PROVIDER == "bedrock":
        return f"bedrock:{BEDROCK_MODEL_ID} ({BEDROCK_REGION})"
    return f"ollama:{OLLAMA_MODEL}"


def get_bedrock_embedding(text: str) -> list[float]:
    """Get embedding from Amazon Bedrock"""
    """ Ok quick function skip model, enviroment, etc.."""
    try:
        # Example with Titan Embeddings
        response = _client().invoke_model(
            modelId="amazon.titan-embed-text-v2:0",
            body=json.dumps(
                {
                    "inputText": text,
                    "dimensions": 512,
                    "normalize": True,
                }
            ),
        )

        response_body = json.loads(response["body"].read())
        return response_body["embedding"]
    except Exception as e:
        print(f"Bedrock API error: {e}")
        raise
