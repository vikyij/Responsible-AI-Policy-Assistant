from src.vector_store import retrieve_chunks
from src.chat_model import generate_answer
from src.embeddings import create_embedding


def answer_question(question):
    retrieved_chunks = retrieve_chunks(
        question=question,
        create_embedding=create_embedding,
        top_k=5
    )

    context = "\n\n".join([
        f"Source {i + 1} | Page {item['chunk']['page']}:\n{item['chunk']['text']}"
        for i, item in enumerate(retrieved_chunks)
    ])

    answer = generate_answer(question, context)

    sources = [
        {
            "page": item["chunk"]["page"],
            "document": item["chunk"]["document"],
            "text": item["chunk"]["text"],
            "score": item["score"]
        }
        for item in retrieved_chunks
    ]

    return answer, sources