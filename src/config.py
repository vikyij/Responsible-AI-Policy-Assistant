import os
from dotenv import load_dotenv


load_dotenv()


APP_NAME = "Responsible AI Policy Assistant API"

LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "800"))
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").strip().lower()

if LLM_PROVIDER == "ollama":
    LLM_BASE_URL = os.getenv(
        "OLLAMA_BASE_URL",
        "http://localhost:11434/v1",
    )
    LLM_API_KEY = "ollama"
    LLM_MODEL = os.getenv(
        "OLLAMA_MODEL",
        "qwen2.5:7b",
    )
elif LLM_PROVIDER == "groq":
    LLM_BASE_URL = os.getenv(
        "GROQ_BASE_URL",
        "https://api.groq.com/openai/v1",
    )
    LLM_API_KEY = os.getenv("GROQ_API_KEY")
    LLM_MODEL = os.getenv(
        "GROQ_MODEL",
        "llama-3.3-70b-versatile",
    )

    if not LLM_API_KEY:
        raise RuntimeError("GROQ_API_KEY is required when LLM_PROVIDER=groq.")
else:
    raise ValueError(
        f"Unsupported LLM provider: {LLM_PROVIDER}"
    )


PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000,https://ai-policy-assistant-1l1a.vercel.app",
    ).split(",")
    if origin.strip()
]
