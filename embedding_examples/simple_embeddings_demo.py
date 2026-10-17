"""
Very basic demonstration of embeddings with caching
Local models needs way too much time to download.
Use bedrock for now with smart caching(csv file) to avoid API spam.
"""

import csv
import json
import os
import sys

import boto3
import numpy as np

# Cache file location
CACHE_FILE = os.path.join(os.path.dirname(__file__), "embeddings_cache.csv")


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
                    "dimensions": 512,  # 256, 512, 1024
                    "normalize": True,
                }
            ),
        )

        response_body = json.loads(response["body"].read())
        return response_body["embedding"]
    except Exception as e:
        print(f"Bedrock API error: {e}")
        raise


def load_cached_embeddings() -> dict:
    """Load embeddings from cache file.
    File should not be empty
    """
    cache = {}
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                if len(row) >= 2:
                    text = row[0]
                    # Convert string representation back to list of floats
                    try:
                        embedding = [float(x) for x in row[1:] if x]
                        cache[text] = embedding
                    except ValueError:
                        print(f"Warning: Could not parse embedding for '{text[:30]}...'")
    return cache


def save_embedding_to_cache(text: str, embedding: list[float]) -> None:
    """Save embedding to cache file"""

    # Create file with header if it doesn't exist
    file_exists = os.path.exists(CACHE_FILE)

    with open(CACHE_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(["text"] + [f"dim_{i}" for i in range(len(embedding))])

        # Write the text and embedding values
        writer.writerow([text] + embedding)


def get_embedding_with_cache(text: str) -> list[float]:
    """Get embedding with caching - first check cache, then generate if needed"""
    # Load existing cache
    cache = load_cached_embeddings()

    # Check if we already have this embedding
    if text in cache:
        print(f"Using cached embedding for: '{text[:40]}...'")
        return cache[text]

    # Generate new embedding
    print(f" Generating new embedding for: '{text[:40]}...'")
    try:
        embedding = get_bedrock_embedding(text)
        # Save to cache
        save_embedding_to_cache(text, embedding)
        return embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        raise


def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors"""
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


if __name__ == "__main__":
    texts = [
        "Microsoft has cybersecurity risks in 2023",
        "Abode got hacked. Multilty vulbunaries were found in the backend services that allowed access without credentials",
        "Apple faces iPhone memory shortage problems in 2026",
        "Google encounters regulatory challenges in Europe",
        "The weather is sunny today",
        "I like eating pizza for dinner",
    ]

    print("AWS Bedrock Embeddings Demo with Caching")
    embeddings = []

    for text in texts:
        try:
            embedding = get_embedding_with_cache(text)
            embeddings.append(embedding)
            print(f"Embedding ready (dimension: {len(embedding)})")
        except Exception as e:
            print(f"Error with embedding: {e}")
            sys.exit(1)

    # Calculate similarities between texts
    print("\nCalculating similarities (cosine similarity)...")

    # Compare with a reference query
    reference_text = "cybersecurity attacks"
    reference_emb = get_embedding_with_cache(reference_text)

    print(f"\n\n\nComparing with: '{reference_text}'\n\n")

    # Calculate all similarities first
    similarities = []
    for i, (text, emb) in enumerate(zip(texts, embeddings)):
        similarity = cosine_similarity(reference_emb, emb)
        similarities.append((text, similarity))

    # highest similarity first
    similarities.sort(key=lambda x: x[1], reverse=True)

    for text, similarity in similarities:
        print(f"  Similarity with '{text}': {similarity:.4f}")

    print(f"Embeddings cached in: {CACHE_FILE}")
