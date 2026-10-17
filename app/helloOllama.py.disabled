import ollama

resp = ollama.chat(
    model="qwen2.5-coder:1.5b",
    messages=[{"role": "user", "content": "Explain RAG in two sentences."}],
)

print(resp.message.content)