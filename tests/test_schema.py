import psycopg


def test_table_schema_is_correct():
    """Test that the company risks table is created properly"""
    DB_URL = "postgresql://myuser:mypassword@localhost:5432/fintech"

    with psycopg.connect(DB_URL, autocommit=True) as connection:
        # Check if table exists
        result = connection.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'company_risks'
            );
        """).fetchone()

        assert result is not None, "Query should return a result"
        assert result[0] == True, "company_risks table should exist"


def test_company_risks_table_structure():
    """Test that the company risks table has the correct structure"""
    DB_URL = "postgresql://myuser:mypassword@localhost:5432/fintech"

    with psycopg.connect(DB_URL, autocommit=True) as connection:
        # Check columns existence
        result = connection.execute("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = 'company_risks'
            ORDER BY ordinal_position
        """).fetchall()

        columns = [row[0] for row in result]
        expected_columns = ["id", "ticker_symbol", "description", "risk_category", "year", "created_at", "updated_at"]

        for col in expected_columns:
            assert col in columns, f"Column {col} should exist in company_risks table"


# Test for just local
# def test_insert_and_search_risks():
#     """Test inserting and searching for risk data"""
#     # Insert test data
#     risk_id = insert_risk_data(
#         ticker_symbol="GOOGL",
#         description="Google faces regulatory risks in Europe regarding antitrust violations.",
#         risk_category="Regulatory",
#         year=2026,
#     )

#     assert risk_id > 0, "Should return a valid ID"

#     # Search by ticker
#     results = search_risks_by_ticker("GOOGL")
#     assert len(results) >= 1, "Should find at least one risk for GOOGL"

#     googl_risk = results[0]
#     assert googl_risk["ticker_symbol"] == "GOOGL"
#     assert "regulatory risks" in googl_risk["description"].lower()
#     assert googl_risk["risk_category"] == "Regulatory"
#     assert googl_risk["year"] == 2026

#     # Search by category
#     results = search_risks_by_category("Regulatory")
#     assert len(results) >= 1, "Should find risks in Regulatory category"

#     # Get all risks
#     all_risks = get_all_risks()
#     assert len(all_risks) >= 1, "Should have at least one risk record"


# def test_search_nonexistent_ticker():
#     """Test searching for a ticker that doesn't exist"""
#     results = search_risks_by_ticker("NONEXISTENT")
#     assert len(results) == 0, "Should return empty list for non-existent ticker"
