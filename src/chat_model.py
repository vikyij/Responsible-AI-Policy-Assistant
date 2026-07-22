from openai import OpenAI
from src.config import LLM_MAX_TOKENS, LLM_BASE_URL, LLM_MODEL, LLM_API_KEY


client = OpenAI(
    base_url=LLM_BASE_URL,
    api_key=LLM_API_KEY
)


def generate_answer(question, context):
    system_prompt = """
        You are a Responsible AI Policy Assistant.

        Answer the user's question using only the context provided.
        If the answer is not in the context, say: "I could not find enough information in the document."
        If evidence is limited, say the coverage is weak or missing. Do not invent evidence.
        """
    
    user_prompt = f"""
        Context: {context}
        Question: {question}
    """


    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=LLM_MAX_TOKENS
    )

    return response.choices[0].message.content
