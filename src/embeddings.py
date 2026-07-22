from huggingface_hub import InferenceClient
from src.config import EMBEDDING_MODEL, HF_TOKEN


client = InferenceClient(
    provider="hf-inference",
    api_key=HF_TOKEN,
)

def create_embedding(text):
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN is required to create embeddings with Hugging Face.")

    if not text or not text.strip():
        raise ValueError("Text is required to create an embedding.")

    embedding = client.feature_extraction(
        text,
        model=EMBEDDING_MODEL,
    )

    if hasattr(embedding, "tolist"):
        embedding = embedding.tolist()

    if embedding and isinstance(embedding[0], list):
        embedding = embedding[0]

    return [float(value) for value in embedding]
