from dotenv import load_dotenv
# from huggingface_hub import InferenceClient
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

# client = InferenceClient(
#     api_key=os.getenv("HF_TOKEN")
# )


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
        model="qwen2.5:7b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=500
    )

    return response.choices[0].message.content