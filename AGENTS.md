# Agent Instructions for Fintech RAG/MCP Learning Project

This project demonstrates financial assistant capabilities using LLMs with tool calling, and explores RAG concepts with PostgreSQL/pgvector.

## Key Components

- `app/agent.py`: Main finance assistant that uses LLMs with tool calling
- `app/llm.py`: Abstraction layer for AWS Bedrock provider (Ollama support planned)
- `app/rag.py`: RAG implementation with PostgreSQL/pgvector
- `app/hello*.py`: Simple examples for each LLM provider
- `database/`: Database-related files including schema definitions
- `tests/`: Pytest-based unit tests

## Essential Commands

### Development Setup

```bash
# Create and activate virtual environment (recommended)
source .venv/bin/activate

# Install required dependencies
pip install -r requirements.txt

# Alternative: Install packages individually
# pip install ollama boto3 pydantic psycopg[binary] psutil pytest

# Start required services (PostgreSQL with pgvector)
# Build the database container (from inside the database/ folder)
cd database && docker build -t fintech-db .

# Run the database container (data will be lost when container stops)
# Pass database credentials as environment variables
docker run --rm -d -p 5432:5432 \
  -e POSTGRES_DB=fintech \
  -e POSTGRES_USER=myuser \
  -e POSTGRES_PASSWORD=mypassword \
  --name fintech-database fintech-db
```

After starting the database, you need to update the risk embeddings for the RAG functionality to work properly:

```bash
# Run the embedding update script
python app/rag.py
```

### Testing
```bash
python -m pytest tests/
```
The above also test DB schema so DB must running

## Environment Configuration

### LLM Providers
Switch between LLM providers using environment variables:
API bedrock header key is already exported at the console most of the time.

```bash
# AWS Bedrock (default)
export LLM_PROVIDER=bedrock
export BEDROCK_MODEL_ID=amazon.nova-lite-v1:0

# Alternative Bedrock models (require appropriate AWS quotas):
# export BEDROCK_MODEL_ID=openai.gpt-oss-20b-1:0

# Ollama support (planned for future implementation)
# export LLM_PROVIDER=ollama
# export OLLAMA_MODEL=qwen2.5:7b
```

Note: Requires AWS account with Bedrock access and quota for the specified models.

### RAG Database Connection
The RAG component connects to PostgreSQL:
```
postgresql://myuser:mypassword@localhost:5432/fintech
```

To stop the database container and clean up all data:
```bash
docker stop fintech-database
```

## Project-Specific Patterns

Do not write unit tests, except if you are asked.

### Agent Architecture
1. Tools are defined as simple Python functions with typed arguments
2. Each tool has a corresponding Pydantic model for validation
3. Tools are registered in the `TOOLS`, `REGISTRY`, and `ARG_MODELS` collections
4. Tool specifications are built dynamically and passed to LLM providers
5. Structured responses are enforced using JSON schema validation

### LLM Abstraction
- `llm.chat()` handles regular conversations with tool calling
- `llm.chat_structured()` enforces responses matching a specific JSON schema
- Provider differences abstracted behind common interfaces
- Automatic conversion between provider-specific formats

### RAG Implementation
- Uses PostgreSQL with pgvector extension
- Connects to local database via Docker container (no persistence)
- Extension setup handled automatically
- Schema initialized via SQL script during container startup
- Embeddings generated using Amazon Bedrock Titan model
- Requires manual update of risk embeddings after database initialization using `python app/rag.py`
