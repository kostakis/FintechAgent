-- Database schema for fintech risk data

-- Table for storing company risk information
CREATE TABLE IF NOT EXISTS company_risks (
    id SERIAL PRIMARY KEY,
    ticker_symbol TEXT NOT NULL,
    description TEXT NOT NULL,
    risk_category TEXT,
    year INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_company_risks_ticker ON company_risks(ticker_symbol);
CREATE INDEX IF NOT EXISTS idx_company_risks_year ON company_risks(year);
CREATE INDEX IF NOT EXISTS idx_company_risks_category ON company_risks(risk_category);

-- Postgress extension - Embeddings
CREATE EXTENSION IF NOT EXISTS vector;


INSERT INTO company_risks (ticker_symbol, description, risk_category, year) 
VALUES
('MSFT', 'Microsoft faces significant cybersecurity challenges in 2023 with multiple reported breaches affecting Azure cloud services. The company is investing heavily in security infrastructure and has hired additional cybersecurity experts to address these vulnerabilities. These incidents have led to increased scrutiny from regulators and impacted customer trust in their cloud offerings.', 'Cybersecurity', 2023),
('GOOGL', 'Google is experiencing regulatory pressure in the European Union due to antitrust concerns and potential fines exceeding $10 billion. The company faces investigations into its advertising practices and dominance in search markets. European regulators are considering measures that could significantly impact Google revenue streams and market position.', 'Regulatory', 2026),
('AAPL', 'Apple global supply chain has been disrupted due to ongoing semiconductor shortages and manufacturing delays in Asia. The company is working with alternative suppliers and has accelerated investment in vertical integration strategies. These supply constraints have affected product launches and increased costs for key components.', 'Supply Chain', 2026),
('AMZN', 'Amazon is navigating complex regulatory environments as governments worldwide scrutinize its marketplace practices and labor conditions. The company faces potential legislation that could force structural changes to its business model and pricing strategies. Increased competition from smaller e-commerce platforms is also putting pressure on market share growth.', 'Regulatory', 2026)
ON CONFLICT DO NOTHING;

-- Add the embedding columns (Simulate scenario where db initially did not have the column)
ALTER TABLE company_risks ADD COLUMN embedding VECTOR(512);

-- Add an index also (too much for our small DB)
CREATE INDEX IF NOT EXISTS idx_company_risks_embedding ON company_risks USING ivfflat (embedding vector_cosine_ops);
