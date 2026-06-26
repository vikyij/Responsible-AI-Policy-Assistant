import faiss
import numpy as np

embedding_dimension = 384

index = faiss.IndexFlatIP(embedding_dimension)

stored_chunks = []


def store_chunks(chunks, document_name):
    vectors = []

    for i, chunk in enumerate(chunks):
        vectors.append(chunk["embedding"])

        stored_chunks.append({
            "id": f"{document_name}-{i}",
            "text": chunk["text"],
            "page": chunk["page"],
            "document": document_name
        })

    vectors = np.array(vectors).astype("float32")

    index.add(vectors)

def retrieve_chunks(question, create_embedding, top_k):
    question_embedding = create_embedding(question)
    question_vector = np.array([question_embedding]).astype("float32")

    scores, indices = index.search(question_vector, top_k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx != -1:
            results.append({
                "score": float(score),
                "chunk": stored_chunks[idx]
            })
    return results
