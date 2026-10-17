"""
Demo showing how embeddings could work with risk data
Using AWS Bedrock with Titan model
No caching here, just for showcase be careful API costs
"""

import json

import boto3
import numpy as np


def get_bedrock_embedding(text: str) -> list:
    """Get embedding from Amazon Bedrock"""
    client = boto3.client("bedrock-runtime", region_name="eu-north-1")

    try:
        # Example with Titan Embeddings
        response = client.invoke_model(
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
        print(f"    Bedrock API error: {e}")
        raise


def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors"""
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# Sample risk data - This will be similar what our DB will have
risks = [
    {"ticker": "MSFT", "description": "Microsoft has risk in 2023 for cybersecurity.", "category": "Cybersecurity", "year": 2023},
    {
        "ticker": "AAPL",
        "description": "Apple in 2026 has risk in iPhones due to memory shortages.",
        "category": "Supply Chain",
        "year": 2026,
    },
    {
        "ticker": "GOOGL",
        "description": "Google faces regulatory risks in Europe regarding antitrust violations.",
        "category": "Regulatory",
        "year": 2026,
    },
]

print("Generating embeddings for risk descriptions...")
risk_embeddings = []
for risk in risks:
    embedding = get_bedrock_embedding(risk["description"])
    risk_embeddings.append(embedding)
    print(f"{risk['ticker']}: {risk['description']}")
    print(f"Embedding generated (dimension: {len(embedding)})\n")


# Function to find similar risks
def find_similar_risks(query, top_k=2):
    print(f"Finding risks similar to: '{query}'")

    # Generate embedding for query
    query_embedding = get_bedrock_embedding(query)

    # Calculate similarities
    similarities = []
    for i, risk in enumerate(risks):
        # Cosine similarity
        sim = cosine_similarity(query_embedding, risk_embeddings[i])
        similarities.append((risk, sim))

    # Sort by similarity (highest first)
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Return top K results
    return similarities[:top_k]


print("SIMILARITY SEARCH EXAMPLES\n")

# Example 1: Query about cybersecurity
results = find_similar_risks("What companies have cybersecurity concerns?")
# print("\nQuery: 'What companies have cybersecurity concerns?'")
for risk, score in results:
    print(f"  {score:.4f} - {risk['ticker']}: {risk['description']}")
print("\n")

# Example 2: Query about supply chain issues
results = find_similar_risks("Are there any supply chain problems?")
# print("\nQuery: 'Are there any supply chain problems?'")
for risk, score in results:
    print(f"  {score:.4f} - {risk['ticker']}: {risk['description']}")
print("\n")

# Example 3: Query about European regulations
results = find_similar_risks("European regulatory challenges for tech companies")
# print("\nQuery: 'European regulatory challenges for tech companies'")
for risk, score in results:
    print(f"  {score:.4f} - {risk['ticker']}: {risk['description']}")
print("\n")
