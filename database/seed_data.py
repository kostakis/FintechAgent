"""
Simple script to insert sample data into the DB.
It should run after container is started.
Just for showcase, also data are inserted in the db init script
"""

import os
import sys

from app.rag import get_db_connection, insert_risk_data

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def seed_sample_data():
    """Sample data for google"""
    connection = get_db_connection()
    risk_id = insert_risk_data(
        connection=connection,
        ticker_symbol="GOOGL",
        description="Google faces regulatory risks in Europe for monopoly strategies.",
        risk_category="Regulatory",
        year=2026,
    )

    print(f"Inserted google data {risk_id}")
    connection.commit()
    connection.close()


if __name__ == "__main__":
    seed_sample_data()
