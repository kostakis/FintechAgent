import json
import logging
import re
from typing import Any


def log_usage(step: int, reply, seconds: float, logger: logging.Logger | None = None) -> None:
    msg = "[step %s] tokens in=%s out=%s time=%.1fs"
    args = (step, reply.input_tokens, reply.output_tokens, seconds)

    if logger is None:
        print(msg % args)
    else:
        logger.info(msg, *args)


def log_usage_bedrock(step: int, reply, seconds: float, logger: logging.Logger | None = None) -> None:
    usage = reply["usage"]
    msg = "[step %s] tokens in=%s out=%s time=%.1fs"
    args = (step, usage["inputTokens"], usage["outputTokens"], seconds)

    if logger is None:
        print(msg % args)
    else:
        logger.info(msg, *args)


def pretty_print_json(data: Any, indent: int = 2) -> str:
    """Return data as indented JSON text. Never raises."""
    try:
        if hasattr(data, "model_dump"):
            data = data.model_dump(exclude_none=True)

        if isinstance(data, (bytes, bytearray)):
            data = data.decode("utf-8", errors="replace")

        if isinstance(data, str):  # JSON text, maybe fenced
            text = data.strip()
            m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
            if m:
                text = m.group(1).strip()
            data = json.loads(text)

        return json.dumps(data, indent=indent, ensure_ascii=False, default=str)
    except json.JSONDecodeError as e:
        return f"[pretty_print_json: not valid JSON ({e})]\n{data}"
    except Exception as e:
        return f"[pretty_print_json: unexpected error ({e})]"
