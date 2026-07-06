import os
from dotenv import load_dotenv
from pinecone import Pinecone
from src.embeddings import create_embedding


load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index_name = os.getenv("PINECONE_INDEX_NAME")
index=pc.Index(index_name)

APP_NAMESPACE = "current_document"

def store_chunks(chunks, document_name):
    vectors = []

    for i, chunk in enumerate(chunks):
        vectors.append({
            "id": f"{document_name}-{i}",
            "values": chunk["embedding"],
            "metadata": {
                "text": chunk["text"],
                "page": chunk["page"],
                "document": document_name
            }
        })


    index.upsert(vectors=vectors, namespace=APP_NAMESPACE)

def retrieve_chunks(question, top_k):
    question_embedding = create_embedding(question)

    results = index.query(
        vector=question_embedding,
        top_k=top_k,
        include_metadata=True,
        namespace=APP_NAMESPACE
    )

    retrieved_chunks = []

    for match in results["matches"]:
        retrieved_chunks.append({
            "score": float(match["score"]),
            "chunk": {
                "text": match["metadata"]["text"],
                "page": match["metadata"]["page"],
                "document": match["metadata"]["document"]
            }
        })

    return retrieved_chunks

def reset_store():
    try:
        index.delete(delete_all=True, namespace=APP_NAMESPACE)
    except Exception:
        pass
