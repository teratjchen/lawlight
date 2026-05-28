import io
import json
import os
from pathlib import Path

import anthropic
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

app = FastAPI(title="Lexible")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load system prompt from prompt.txt — edit that file to change the AI's behavior.
SYSTEM_PROMPT = Path("prompt.txt").read_text()

# How responses are tailored per persona — added to each request (not the cached prompt).
PERSONA_INSTRUCTIONS = {
    "novice": (
        "USER KNOWLEDGE LEVEL: Complete beginner — no legal background whatsoever. "
        "Write at a 5th-grade reading level. Use zero jargon; if a legal term is unavoidable, "
        "immediately explain it in parentheses using everyday words. "
        "Use relatable analogies (e.g. 'think of this like a bill from a store'). "
        "Be warm, clear, and reassuring in tone — avoid anything that feels intimidating. "
        "In the glossary, define even basic-sounding terms. "
        "In rights_under_law, be concrete about what this means for an everyday person. "
        "Include all available local resources."
    ),
    "intermediate": (
        "USER KNOWLEDGE LEVEL: General public with a basic understanding of how legal systems work. "
        "Write in plain English. You may use common legal terms (like 'eviction', 'defendant', 'statute') "
        "but briefly explain any specialized or technical language. "
        "Balanced depth — thorough but not overwhelming. Standard tone."
    ),
    "expert": (
        "USER KNOWLEDGE LEVEL: Legal professional, law student, or highly experienced person. "
        "You may use full legal terminology without lay explanations. "
        "Provide deeper statutory analysis and more case law citations. "
        "Plain summaries can be concise — prioritize legal precision over simplicity. "
        "Include procedural nuances and any minority or split-jurisdiction considerations where relevant."
    ),
}


class AnalyzeRequest(BaseModel):
    text: str
    language: str = "English"
    persona: str = "intermediate"


@app.get("/")
async def root():
    return FileResponse("index.html")


@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="Server configuration error: ANTHROPIC_API_KEY is not set."
        )

    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="No document text provided.")
    if len(text) > 60_000:
        raise HTTPException(
            status_code=400,
            detail="Document too long (max ~40 pages). Try pasting a specific section."
        )

    persona_ctx = PERSONA_INSTRUCTIONS.get(req.persona, PERSONA_INSTRUCTIONS["intermediate"])
    user_msg = (
        f"{persona_ctx}\n\n"
        f"Respond entirely in {req.language}.\n\n"
        f"DOCUMENT:\n{text}"
    )

    client = anthropic.Anthropic(api_key=api_key)

    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user_msg}],
        )
        raw = response.content[0].text.strip()
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError("No JSON found in response")
        result = json.loads(raw[start:end])
        return JSONResponse(content=result)

    except (json.JSONDecodeError, ValueError) as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse response: {e}")
    except anthropic.APIError as e:
        raise HTTPException(status_code=502, detail=f"AI service error: {e}")


@app.post("/extract-pdf")
async def extract_pdf(file: UploadFile = File(...)):
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF.")
    try:
        import pdfplumber
    except ImportError:
        raise HTTPException(status_code=500, detail="PDF support unavailable on this server.")

    try:
        content = await file.read()
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
        text = "\n\n".join(pages).strip()
        if not text:
            raise HTTPException(
                status_code=400,
                detail="Could not extract text — PDF may be a scanned image. Try copying and pasting instead."
            )
        return {"text": text[:60_000], "pages": len(pages)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read PDF: {e}")
