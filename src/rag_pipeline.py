from src.vector_store import retrieve_chunks
from src.chat_model import generate_answer


RESPONSIBLE_AI_CATEGORIES = [
    "fairness and bias mitigation",
    "transparency and explainability",
    "accountability and ownership",
    "privacy and data protection",
    "human oversight",
    "security and robustness",
    "monitoring after deployment",
    "incident response",
    "documentation and auditability",
    "user rights, appeals, or contestability"
]

def retrieve_context(query, top_k=5):
    retrieved_chunks = retrieve_chunks(
        question=query,
        top_k=top_k
    )

    context = "\n\n".join([
        f"Source {i + 1} | Page {item['chunk']['page']}:\n{item['chunk']['text']}"
        for i, item in enumerate(retrieved_chunks)
    ])

    sources = [
        {
            "page": item["chunk"]["page"],
            "document": item["chunk"]["document"],
            "text": item["chunk"]["text"],
            "score": item["score"]
        }
        for item in retrieved_chunks
    ]

    return context, sources

def answer_question(question):
        context, sources = retrieve_context(question, top_k=5)
        answer = generate_answer(question, context)

        return answer, sources

def retrieve_context_by_categories(categories, top_k=3):
    all_context_parts = []
    all_sources = []

    for category in categories:
        query=f"What does the document say about {category}?"

        context, sources = retrieve_context(query, top_k)

        all_context_parts.append(f"=== {category.upper()} ===\n{context}")
        all_sources.extend(sources)
   
    full_context = "\n\n".join(all_context_parts)
    return full_context, all_sources
     

def generate_responsible_ai_checklist():
    full_context, all_sources = retrieve_context_by_categories(RESPONSIBLE_AI_CATEGORIES, top_k=3)

    checklist_question= """

    Generate a concise Responsible AI coverage checklist from this document.

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
    - Evidence from the document: one short sentence explaining the coverage

    Do not include a title, introduction, summary, or conclusion.
    Start directly with the first category: "1. Fairness and bias".
    Do not include recommendations. Keep each category brief so the checklist can be scanned in 30 seconds.
    """
    
    answer = generate_answer(checklist_question, full_context)
    return answer, all_sources

def perform_gap_analysis():
    full_context, all_sources = retrieve_context_by_categories(RESPONSIBLE_AI_CATEGORIES, top_k=3)

    question = """
      Perform a Responsible AI gap analysis using the provided document evidence.

        Return your answer in the following format:

        ### 1. Fairness and Bias Mitigation
            - Coverage:
            - Evidence:
            - Gap Identified:
            - Recommendation:

        ### 2. Transparency and Explainability
        ...

        Repeat this structure for all categories.

        Do not output a blank template.
        Do not repeat the field names before the analysis begins.
        Start immediately with the first category.

        End with:
            - Overall assessment
            - Top 3 most important gaps to fix.
        """

    answer = generate_answer(question, full_context)

    return answer, all_sources
