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
