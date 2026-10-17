# Fintech Risk Analysis Assistant

A financial assistant powered by Large Language Models (LLMs) with Retrieval-Augmented Generation (RAG) capabilities for analyzing company risks. This project demonstrates advanced AI techniques including tool calling, semantic search, and multi-provider LLM support.

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## Features

- **Multi-Provider LLM Support**: Seamlessly switch between Ollama (local) and AWS Bedrock (cloud) providers
- **Tool Calling**: Integrated financial tools for stock prices, calculations, and risk analysis
- **RAG Implementation**: Semantic search capabilities using PostgreSQL with pgvector for contextual risk analysis
- **Structured Output**: Enforces consistent JSON responses through schema validation
- **Interactive Chat Interface**: Conversational interface for querying financial risks
- **Database Integration**: PostgreSQL backend with sample risk data for major tech companies

## Project Overview

This financial assistant specializes in risk analysis for publicly traded companies. Users can ask questions about company risks, and the assistant will:
1. Search for relevant risks using semantic similarity
2. Retrieve detailed risk information when needed
3. Provide structured, contextual responses with confidence ratings

### Key Components

- `app/agent.py`: Main finance assistant with tool calling orchestration
- `app/llm.py`: Abstraction layer supporting Ollama and AWS Bedrock providers
- `app/rag.py`: RAG implementation with PostgreSQL/pgvector for semantic search
- `database/`: Database schema and sample risk data for major tech companies
- `tests/`: Comprehensive test suite covering functionality and schema validation

## Prerequisites

- Python 3.8+
- Docker (for database)
- Ollama (for local LLM) OR AWS account with Bedrock access

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/fintech-risk-assistant.git
cd fintech-risk-assistant
```

### 2. Set Up Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the Database

```bash
# Build the database container
cd database
docker build -t fintech-db .
cd ..

# Run the database container
docker run --rm -d -p 5432:5432 \
  -e POSTGRES_DB=fintech \
  -e POSTGRES_USER=myuser \
  -e POSTGRES_PASSWORD=mypassword \
  --name fintech-database fintech-db
```

### 5. Run the Assistant

With Ollama (default):
```bash
python app/agent.py
```

With AWS Bedrock:
```bash
LLM_PROVIDER=bedrock python app/agent.py
```

Example questions to ask:
- "Which companies face cybersecurity risks?"
- "Are there any supply chain issues in the database?"
- "What regulatory challenges is Google facing?"

## Architecture

### Agent Design
1. Tools are defined as simple Python functions with typed arguments
2. Each tool has a corresponding Pydantic model for validation
3. Tools are registered in centralized collections (`TOOLS`, `REGISTRY`, `ARG_MODELS`)
4. Tool specifications are built dynamically and passed to LLM providers
5. Structured responses are enforced using JSON schema validation

### LLM Abstraction
- Unified `llm.chat()` for regular conversations with tool calling
- Unified `llm.chat_structured()` for schema-enforced responses
- Provider differences abstracted behind common interfaces
- Automatic conversion between provider-specific formats

### RAG Implementation
- Uses PostgreSQL with pgvector extension for semantic search
- Connects to local database via Docker container
- Embeddings generated using Amazon Bedrock Titan model
- Semantic similarity search with relevance scoring

## Sample Data

The database includes sample risk data for major tech companies:

| Company | Ticker | Risk Categories |
|---------|--------|----------------|
| Microsoft | MSFT | Cybersecurity |
| Google | GOOGL | Regulatory |
| Apple | AAPL | Supply Chain |
| Amazon | AMZN | Regulatory |

## Configuration

### LLM Providers

Switch between LLM providers using environment variables:

```bash
# Ollama (default)
export LLM_PROVIDER=ollama
export OLLAMA_MODEL=qwen2.5:7b

# AWS Bedrock
export LLM_PROVIDER=bedrock
export BEDROCK_REGION=us-east-1
export BEDROCK_MODEL_ID=amazon.nova-lite-v1:0
```

### Database Connection

The RAG component connects to PostgreSQL using:
```
postgresql://myuser:mypassword@localhost:5432/fintech
```

## Development

### Running Tests

```bash
# Ensure database is running first
python -m pytest tests/
```

### Project Structure

```
├── app/
│   ├── agent.py          # Main assistant logic
│   ├── llm.py            # LLM abstraction layer
│   ├── rag.py            # RAG implementation
│   ├── helloBedrock.py   # Bedrock example
│   ├── helloOllama.py    # Ollama example
│   └── utils.py          # Utility functions
├── database/
│   ├── Dockerfile        # Database container definition
│   ├── schema.sql        # Database schema
│   └── seed_data.py      # Sample data generator
├── tests/
│   ├── test_agent.py     # Agent functionality tests
│   └── test_schema.py    # Database schema tests
└── requirements.txt      # Python dependencies
```