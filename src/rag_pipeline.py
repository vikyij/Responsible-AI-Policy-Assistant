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

def generate_responsible_ai_checklist():
    checklist_question= """

    Generate a Responsible AI checklist from this document.

    Assess the document under these categories:
    1. Fairness and bias
    2. Transparency and explainability
    3. Accountability and ownership
    4. Privacy and data protection
    5. Human oversight
    6. Security and robustness
    7. Monitoring after deployment
    8. Incident response

    For each category, return:
    - Status: Covered, Partially covered, or Missing
    - Evidence from the document
    - Recommendation
    """
    return answer_question(checklist_question)
