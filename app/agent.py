import inspect
import json
import logging
import logging.config
import os
import re
import time
from pathlib import Path
from typing import Literal

import rag
from utils import log_usage, pretty_print_json

BASE_DIR = Path(__file__).parent


def setup_logging():
    try:
        (BASE_DIR / "logs").mkdir(exist_ok=True)  # file handler won't create the folder
        print(f"Search logging ini file at {BASE_DIR}")
        logging.config.fileConfig(BASE_DIR / "logging.ini", disable_existing_loggers=False)
    except Exception:
        # Fallback to basic config for tests/simple runs
        logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")


logger = logging.getLogger("agent")

import llm
from pydantic import BaseModel, Field, ValidationError

VERBOSE = os.getenv("AGENT_VERBOSE", "1") == "1"


def get_price(ticker: str) -> dict:
    """Get the latest stock price for a ticker symbol.

    Args:
        ticker: Stock ticker symbol, for example AAPL.

    Returns:
        dict: The ticker and its latest price.
    """
    fake = {"AAPL": 231.5, "MSFT": 420.1}
    return {"ticker": ticker, "price": fake.get(ticker.upper(), "unknown")}


def calculate(a: float, b: float, op: str) -> dict:
    """Do basic arithmetic on two numbers.

    Args:
        a: First number.
        b: Second number.
        op: One of "add", "sub", "mul", "div".

    Returns:
        dict: The result of the operation.
    """
    ops = {
        "add": a + b,
        "sub": a - b,
        "mul": a * b,
        "div": a / b if b else "division by zero",
    }
    return {"result": ops.get(op, "unknown op")}


def search_risks_semantic(query: str, limit: int = 5) -> dict:
    """Search for risks using semantic similarity"""
    try:
        with rag.get_db_connection() as connection:
            results = rag.semantic_search_risks(connection, query, limit * 3)  # Get more results initially

            # Filter results to only include those with similarity >= 0.3
            filtered_results = [r for r in results if r.get("similarity", 0) >= 0.3]
            final_results = filtered_results[:limit]

            # Add relevance interpretation
            for result in final_results:
                sim = result.get("similarity", 0)
                if sim >= 0.5:
                    result["relevance"] = "High"
                elif sim >= 0.3:
                    result["relevance"] = "Medium"
                else:
                    result["relevance"] = "Low"

            # Prepare concise results for LLM context (no descriptions to save tokens)
            results_for_llm_context = []
            for result in final_results:
                results_for_llm_context.append(
                    {
                        "id": result["id"],
                        "ticker": result["ticker_symbol"],
                        "risk_category": result["risk_category"],
                        "similarity": round(result["similarity"], 4),
                        "relevance": result["relevance"],
                    }
                )

            if len(final_results) == 0:
                msg = "No relevant risks found in the database. Do NOT search again."
            elif len(final_results) < limit:
                msg = (
                    f"Only {len(final_results)} matching record(s) exist in the entire database "
                    f"(fewer than the requested {limit}). Do NOT call search_risks_semantic again "
                    "with different phrasing; proceed immediately to answer with the available data."
                )
            else:
                msg = "Search complete. Do NOT search again. Answer using the provided data, or call 'get_risk_details' if full descriptions are needed."

            response = {
                "query": query,
                "results_found": len(final_results),
                "requested_limit": limit,
                "risks_summary": results_for_llm_context,
                "interpretation": "Similarity scores: 1.0=Identical, 0.5=Highly Relevant, 0.3=Moderately Relevant",
                "search_complete": True,
                "message": msg,
            }

            return response
    except Exception as e:
        return {"error": str(e), "risks": []}


def get_risk_details(risk_id: int) -> dict:
    """Get detailed information about a specific risk by database ID.

    Use this when you need the complete description of a risk that was previously identified.
    The risk ID is provided in semantic search results."""
    try:
        with rag.get_db_connection() as connection:
            result = connection.execute(
                """
                SELECT id, ticker_symbol, description, risk_category, year
                FROM company_risks
                WHERE id = %s
                """,
                (risk_id,),
            )

            row = result.fetchone()
            if row:
                return {"id": row[0], "ticker_symbol": row[1], "description": row[2], "risk_category": row[3], "year": row[4]}
            else:
                return {"error": f"No risk found with ID {risk_id}"}
    except Exception as e:
        return {"error": str(e)}


class GetPriceArgs(BaseModel):
    ticker: str = Field(description="Stock ticker symbol, for example AAPL.")


class CalculateArgs(BaseModel):
    a: float = Field(description="First number. Must be a real number, never a placeholder.")
    b: float = Field(description="Second number.")
    op: Literal["add", "sub", "mul", "div"] = Field(description="The operation to apply.")


class SearchRisksSemanticArgs(BaseModel):
    query: str = Field(description="The search query to describe the risk you are looking for.")
    limit: int = Field(default=5, description="Maximum number of results to return")


class GetRiskDetailsArgs(BaseModel):
    risk_id: int = Field(description="Database ID of the risk record to retrieve. Provided in previous search results.")


class FinalAnswer(BaseModel):
    answer: str
    sources: list[str]  # filled in by our code, not trusted from the model
    confidence: Literal["low", "medium", "high"]


# Theese need to be in SYNC ALWAYS
TOOLS = [get_price, calculate, search_risks_semantic, get_risk_details]
REGISTRY = {f.__name__: f for f in TOOLS}
ARG_MODELS = {
    "get_price": GetPriceArgs,
    "calculate": CalculateArgs,
    "search_risks_semantic": SearchRisksSemanticArgs,
    "get_risk_details": GetRiskDetailsArgs,
}


def build_tool_specs() -> list[dict]:
    """Neutral tool descriptions (name, description, JSON schema) for any provider."""
    specs = []
    for f in TOOLS:
        doc = inspect.getdoc(f) or ""
        description = doc.split("\n\n")[0].replace("\n", " ")  # first paragraph
        specs.append(
            {
                "name": f.__name__,
                "description": description,
                "parameters": ARG_MODELS[f.__name__].model_json_schema(),
            }
        )
    return specs


TOOL_SPECS = build_tool_specs()

SYSTEM = (
    "You are a finance assistant specializing in risk analysis. "
    "You can search for financial risks using semantic queries, by ticker symbol, or by risk category. "
    "Call 'search_risks_semantic' AT MOST ONCE per topic. "
    "When you receive a response with 'search_complete: true', you MUST STOP calling 'search_risks_semantic' immediately—even if fewer results were returned than the user asked for (e.g., 1 result found when 2 were requested). "
    "Never rephrase your query to search a second time. If the database has fewer companies than requested, simply report the ones found and state that no other matching companies exist in the database. "
    "If you need the full description of a risk, use 'get_risk_details' with the ID from 'risks_summary', then provide your final answer. "
    "Never guess numbers. "
    "If you need a stock price, call get_price first. "
    "Call one tool at a time. If a call depends on a previous result, wait for it. "
    "Use the tool-calling interface, never write JSON in your reply."
)


def execute_tool(name: str, args: dict) -> dict:
    """Validate the argument of the tool/function and run it"""
    if name not in REGISTRY or name not in ARG_MODELS:
        return {"error": f"Unknown tool '{name}'. Available: {list(REGISTRY)}"}
    try:
        valid = ARG_MODELS[name](**args)  # checks types and allowed values
        return REGISTRY[name](**valid.model_dump())
    except ValidationError as e:
        return {"error": f"Invalid arguments: {e.errors(include_url=False)}"}
    except Exception as e:
        return {"error": str(e)}


def clean_thinking(text: str) -> str:
    """Remove <thinking>...</thinking> tags from the model output."""
    return re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL).strip()


def finalize(messages: list, tools_called: set[str]) -> FinalAnswer | None:
    """Ask for the final answer as validated JSON - Ask for structured chat"""
    last_assistant_reply = clean_thinking(messages[-1].get("content", ""))
    request = {
        "role": "user",
        "content": (
            "Format the assistant's latest reply into JSON with fields: "
            f"answer (use this exact latest reply: {last_assistant_reply!r}, ensuring it directly answers the user's very last message with relevant numbers and units), "
            "sources (list of tool names; any value is fine, it will be overwritten), "
            "confidence (low, medium or high). "
            "Do NOT summarize earlier turns in the conversation."
        ),
    }
    msgs = messages + [request]

    for attempt in range(2):  # one retry
        try:
            raw = llm.chat_structured(msgs, SYSTEM, FinalAnswer.model_json_schema(), TOOL_SPECS)
            logger.debug("Structured LLM response %s", raw)
        except Exception as e:
            logger.exception(
                "Structured LLM call failed (attempt %d/2): %s",
                attempt + 1,
                type(e).__name__,
            )
            return None

        try:
            final = FinalAnswer.model_validate_json(raw)
            final.sources = sorted(tools_called)
            logger.debug("Final response validated and returned to user")

            return final
        except ValidationError as e:
            logger.exception("Response could not be validated")
            msgs = msgs + [
                {"role": "assistant", "content": raw or "(empty)"},
                {
                    "role": "user",
                    "content": f"That was invalid: {e}. Return valid JSON only.",
                },
            ]
    return None


# The agent loop
def run(question: str, messages: list | None = None, max_steps: int = 5) -> tuple[str, list]:
    # Initialize history if none was provided, otherwise append the new question
    if messages is None:
        messages = []
    messages.append({"role": "user", "content": question})

    # Track tools called during THIS turn only
    tools_called: set[str] = set()

    for step in range(max_steps):  # Do not loop forever
        logger.info("Iteration %d sending %d messages", step, len(messages))
        logger.debug("Sending Messages: %s", pretty_print_json(messages))

        start = time.time()
        try:
            logger.info("Sending tools and system message")
            reply = llm.chat(messages, SYSTEM, TOOL_SPECS)
        except Exception as e:
            return f"LLM call failed: {e}", messages

        log_usage(step, reply, time.time() - start, logger)

        # Add the assistant reply to the shared history
        messages.append(
            {
                "role": "assistant",
                "content": reply.text,
                "tool_calls": [{"id": c.id, "name": c.name, "args": c.args} for c in reply.tool_calls],
            }
        )

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Response of iteration %s: %s", step, pretty_print_json(messages[-1]))

        if not reply.tool_calls:  # No tool requested so this is the final answer
            logger.info("Response no tool call, treating it as final response")

            # Fast path: if no tools were called in this turn, skip the extra LLM call
            if not tools_called:
                final = FinalAnswer(
                    answer=clean_thinking(reply.text),
                    sources=[],
                    confidence="high",
                )
                return final.model_dump_json(indent=2), messages

            final = finalize(messages, tools_called)
            if final is None:  # could not get valid JSON, return the plain text
                logger.error("Could not validate response, returning all the text to user anyway")
                return clean_thinking(reply.text), messages
            return final.model_dump_json(indent=2), messages

        for i, call in enumerate(reply.tool_calls):
            if i > 0:
                result = {"error": "Skipped: call one tool at a time and wait for the previous result."}
            else:
                logger.info("step [%d] tool call: %s (%s)", step, call.name, call.args)
                result = execute_tool(call.name, call.args)
                logger.debug("Result of tool call %s", result)
                tools_called.add(call.name)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": call.name,
                    "content": json.dumps(result, default=str),
                }
            )

    return "Stopped: max steps reached.", messages


if __name__ == "__main__":
    setup_logging()
    logger = logging.getLogger("agent")

    logger.info("Starting main")
    logger.info(f"LLM Provider: {llm.describe()}")

    chat_history = []
    print("Finance Risk Agent initialized. Type 'exit' to quit, or 'clear' to reset memory.\n")
    print("You can ask questions like: \n \
        Top companies with cybersecurity risks? \n \
        Any companies with supply chain issues ? \n")

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break
        if user_input.lower() == "clear":
            chat_history.clear()
            print("Conversation history cleared.")
            continue

        answer, chat_history = run(user_input, messages=chat_history)
        print(f"\nAgent:\n{answer}")

# if __name__ == "__main__":
#     setup_logging()
#     logger = logging.getLogger("agent")

#     logger.info("Starting main")
#     logger.info(f"LLM Provider: {llm.describe()}")

#     # first_question_answer = run("What is AAPL's price, and what is it multiplied by 10?")
#     # logger.info(first_question_answer)

#     # second_question_answer = run("What is MSFT's price, and what is it multiplied by 10?")
#     # logger.info(second_question_answer)

#     third_question_answer = run("What are the top 2 companies with the most cybersecurity risks?")
#     logger.info(third_question_answer)

#     # Context is reset so we can not really ask this after
#     logger.info(run("What is the problem with MSFT?"))

#     # third_question_answer = run("Any company facing supply issues? I think there are some memory chip issues going around.")
#     # logger.info(third_question_answer)
