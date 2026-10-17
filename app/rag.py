import json
import logging
import logging.config
from pathlib import Path
from typing import Any

import psycopg
from llm import get_bedrock_embedding

BASE_DIR = Path(__file__).parent


def setup_logging() -> None:
    try:
        (BASE_DIR / "logs").mkdir(exist_ok=True)  # file handler won't create the folder
        print(f"Search logging ini file at {BASE_DIR}")
        logging.config.fileConfig(BASE_DIR / "logging.ini", disable_existing_loggers=False)
    except Exception:
        # Fallback to basic config for tests/simple runs
        logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")


logger = logging.getLogger("rag")


def setup_extension(conn: psycopg.Connection) -> None:
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")


def insert_risk_data(
    connection, ticker_symbol: str, description: str, risk_category: str | None = None, year: int | None = None
) -> int | None:
    """Insert a new risk record into the database

    Args:
        connection: Active database connection object
        ticker_symbol: Stock ticker symbol
        description: Description of the risk
        risk_category: Category of the risk (optional)
        year: Year of the risk (optional)

    Returns:
        ID of the inserted record, or None if insertion failed
    """
    result = connection.execute(
        """
        INSERT INTO company_risks (ticker_symbol, description, risk_category, year)
        VALUES (%s, %s, %s, %s)
        RETURNING id
        """,
        (ticker_symbol, description, risk_category, year),
    )

    record_id = result.fetchone()
    if record_id is None:
        logger.error("Insertion operation failed")
        return None
    else:
        record_id = record_id[0]

    logger.info("Inserted new risk record with ID: %s for ticker: %s", record_id, ticker_symbol)
    return record_id


def search_risks_by_ticker(connection, ticker_symbol: str) -> list[dict[str, Any]]:
    """Find all risks for a specific stock ticker symbol

    This function searches our database and returns a list of risks
    associated with the given ticker symbol.

    Args:
        connection: Active database connection object
        ticker_symbol: Stock ticker like 'AAPL' for Apple or 'GOOGL' for Google

    Returns:
        List of risk dictionaries with id, ticker_symbol, description, risk_category, and year.
        Returns an empty list if no risks are found for the ticker.
    """
    result = connection.execute(
        """
        SELECT id, ticker_symbol, description, risk_category, year
        FROM company_risks
        WHERE ticker_symbol = %s
        ORDER BY year DESC
        """,
        (ticker_symbol,),
    )

    rows = result.fetchall()

    # Convert each row to a dictionary
    risks = []
    for row in rows:
        risk_dict = {"id": row[0], "ticker_symbol": row[1], "description": row[2], "risk_category": row[3], "year": row[4]}
        risks.append(risk_dict)

    return risks


def search_risks_by_category(connection, risk_category: str) -> list[dict[str, Any]]:
    """Find all risks for a specific risk category

    This function searches our database and returns a list of risks
    associated with the given risk category.

    Args:
        connection: Active database connection object
        risk_category: Risk category like 'Regulatory' or 'Operational'

    Returns:
        List of risk dictionaries with id, ticker_symbol, description, risk_category, and year.
        Returns an empty list if no risks are found for the category.
    """
    result = connection.execute(
        """
        SELECT id, ticker_symbol, description, risk_category, year
        FROM company_risks
        WHERE risk_category ILIKE %s
        ORDER BY year DESC
        """,
        (f"%{risk_category}%",),
    )

    rows = result.fetchall()

    # Convert each row to a dictionary
    risks = []
    for row in rows:
        risk_dict = {"id": row[0], "ticker_symbol": row[1], "description": row[2], "risk_category": row[3], "year": row[4]}
        risks.append(risk_dict)

    return risks


def get_all_risks(connection, limit: int = 100) -> list[dict[str, Any]]:
    """Get all risks with optional limit

    This function retrieves all risks from the database, up to the specified limit.

    Args:
        connection: Active database connection object
        limit: Maximum number of risks to retrieve (default: 100)

    Returns:
        List of risk dictionaries with id, ticker_symbol, description, risk_category, and year.
        Returns an empty list if no risks are found.
    """
    result = connection.execute(
        """
        SELECT id, ticker_symbol, description, risk_category, year
        FROM company_risks
        ORDER BY created_at DESC
        LIMIT %s
        """,
        (limit,),
    )

    # Fetch all rows from the result
    rows = result.fetchall()

    # Convert each row to a dictionary
    risks = []
    for row in rows:
        risk_dict = {"id": row[0], "ticker_symbol": row[1], "description": row[2], "risk_category": row[3], "year": row[4]}
        risks.append(risk_dict)

    return risks


def generate_embedding(text: str) -> list[float]:
    try:
        embedding = get_bedrock_embedding(text)
        return embedding
    except Exception as e:
        logger.error(f"Can not generate embedding for {text} {e}")
        # Fallback to a default zero so app does not crash
        return [0.0] * 512  # assums 512 alwats can be made argument


def update_risk_embeddings(connection) -> None:
    """Update embeddings for all existing risk records"""
    result = connection.execute(
        """
        SELECT id, description
        FROM company_risks
        WHERE embedding IS NULL
        """
    )

    rows = result.fetchall()
    for row in rows:
        risk_id, description = row
        embedding = generate_embedding(description)

        connection.execute(
            """
            UPDATE company_risks
            SET embedding = %s
            WHERE id = %s
            """,
            (embedding, risk_id),
        )
    connection.commit()


def get_db_connection():
    """Get a database connection with improved error handling and logging"""
    DB_URL = "postgresql://myuser:mypassword@localhost:5432/fintech"

    # Extract connection info for logging (without password)
    conn_info = "postgresql://myuser:***@localhost:5432/fintech"
    logger.info(f"Attempting to connect to database: {conn_info}")

    try:
        connection = psycopg.connect(DB_URL)
        logger.info("Successfully connected to database")
        return connection
    except psycopg.OperationalError as e:
        logger.error(f"Operational error connecting to database: {conn_info}. Error: {str(e)}")
        raise
    except psycopg.DatabaseError as e:
        logger.error(f"Database error connecting to database: {conn_info}. Error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error connecting to database: {conn_info}. Error: {str(e)}")
        raise


def semantic_search_risks(connection, query: str, limit: int = 5) -> list[dict[str, Any]]:
    """Perform semantic search on risks using embeddings"""

    # Generate embedding for the query
    query_embedding = generate_embedding(query)

    result = connection.execute(
        """
        SELECT id, ticker_symbol, description, risk_category, year, (embedding <=> %s::vector) AS similarity
        FROM company_risks
        WHERE embedding IS NOT NULL
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """,
        (query_embedding, query_embedding, limit),
    )

    results = []
    for row in result:
        results.append(
            {
                "id": row[0],
                "ticker_symbol": row[1],
                "description": row[2],
                "risk_category": row[3],
                "year": row[4],
                "similarity": 1 - row[5],  # Convert distance to similarity score
            }
        )

    #logger.debug(f"All risk records: {json.dumps(results, indent=2)}")
    return results


def main() -> None:
    # Open a connection to the DB
    with get_db_connection() as connection:
        logger.info("Opened connection with database")
        update_risk_embeddings(connection)
        semantic_search_risks(connection, "Any company having supply chain issues?")
        # setup_extension(connection)


# Main just for local examples
if __name__ == "__main__":
    setup_logging()
    logger = logging.getLogger("rag")

    main()
