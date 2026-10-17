import time
from typing import Literal

import boto3
from pydantic import BaseModel, Field
from utils import log_usage_bedrock

client = boto3.client("bedrock-runtime", region_name="eu-north-1")
# model_id = "amazon.nova-lite-v1:0"
model_id = "eu.amazon.nova-2-lite-v1:0"  # EU inference profile

TOOL_CONFIG = {}


start = time.time()

response = client.converse(
    modelId=model_id,
    system=[{"text": "MCP Means Model Context Protocol"}],
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "text": "Explain what is RAG in AI context. How is RAG different from MCP ? \
                    Or they are related closely enough ?. Max 3 sentences response"
                }
            ],
        }
    ],
    # toolConfig=TOOL_CONFIG,
)

endTime = time.time() - start
log_usage_bedrock(0, response, endTime)

print(response["output"]["message"]["content"][0]["text"])


## --- TOOLS -------------------------------------------------------


class GetPriceArgs(BaseModel):
    ticker: str = Field(description="Stock ticker symbol, for example AAPL.")


class CalculateArgs(BaseModel):
    a: float = Field(description="First number. Must be a real number, never a placeholder.")
    b: float = Field(description="Second number.")
    op: Literal["add", "sub", "mul", "div"] = Field(description="The operation to apply.")


# https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ToolSpecification.html
def create_tool_spec(tool_name: str, tool_description: str, arg_model) -> dict:
    return {
        "toolSpec": {
            "name": tool_name,
            "description": tool_description,
            "inputSchema": {"json": arg_model.model_json_schema()},
        }
    }


# https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ToolConfiguration.html#bedrock-Type-runtime_ToolConfiguration-tools
TOOL_CONFIG = {
    "tools": [
        create_tool_spec(
            "get_price",
            "Get the latest stock price for a ticker symbol, e.g. AAPL.",
            GetPriceArgs,
        ),
        create_tool_spec(
            "calculate",
            "Do basic arithmetic. op is one of add, sub, mul, div.",
            CalculateArgs,
        ),
    ]
}
