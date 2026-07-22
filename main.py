from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.chunker import chunk_pages
from src.config import APP_NAME, CORS_ORIGINS, LLM_MODEL, LLM_PROVIDER, PINECONE_INDEX_NAME
from src.embeddings import create_embedding
from src.extract_text import extract_text
from src.rag_pipeline import (
    answer_question,
    generate_responsible_ai_checklist,
    perform_gap_analysis,
)
from src.vector_store import reset_store, store_chunks


SUGGESTED_QUESTIONS = [
    "What does this document say about fairness and bias?",
    "What risks are identified in this document?",
    "Does this document mention human oversight?",
    "What accountability mechanisms are described?",
    "What does this document say about transparency?",
    "What gaps exist in this policy?",
    "What does this document say about privacy?",
]


class Source(BaseModel):
    page: int
    document: str
    text: str
    score: float


class ChatRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    answer: str
    sources: List[Source]


class UploadResponse(BaseModel):
    document: str
    pages_processed: int
    chunks_indexed: int
    status: str
    suggested_questions: List[str]


class HealthResponse(BaseModel):
    status: str
    app: str
    provider: str
    model: str
    pinecone_index: str | None


app = FastAPI(title=APP_NAME)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health_check():
    return {
        "status": "ok",
        "app": APP_NAME,
        "provider": LLM_PROVIDER,
        "model": LLM_MODEL,
        "pinecone_index": PINECONE_INDEX_NAME,
    }


@app.post("/documents/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")

    try:
        reset_store()
        file.file.seek(0)
        pages = extract_text(file.file)
        chunks = chunk_pages(pages)

        for chunk in chunks:
            chunk["embedding"] = create_embedding(chunk["text"])

        store_chunks(chunks, file.filename)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "document": file.filename,
        "pages_processed": len(pages),
        "chunks_indexed": len(chunks),
        "status": "indexed",
        "suggested_questions": SUGGESTED_QUESTIONS,
    }


@app.post("/chat", response_model=AnswerResponse)
def chat(request: ChatRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is required.")

    try:
        answer, sources = answer_question(question)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {"answer": answer, "sources": sources}


@app.post("/checklist", response_model=AnswerResponse)
def checklist():
    try:
        answer, sources = generate_responsible_ai_checklist()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {"answer": answer, "sources": sources}


@app.post("/gap-analysis", response_model=AnswerResponse)
def gap_analysis():
    try:
        answer, sources = perform_gap_analysis()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {"answer": answer, "sources": sources}
