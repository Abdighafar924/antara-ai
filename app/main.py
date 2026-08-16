"""
MediCore AI — FastAPI Backend
Stateless: upload a CSV, get back JSON (KPIs, findings, schema) or a PDF/PPTX report.
No DB required. Designed to sit behind a React frontend and, later, read from
Clinical OS via clinic_api instead of manual CSV upload.
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import io
import os
import requests

from app import core, findings as findings_module, reports

# Groq TTS (Orpheus models) -- key lives only here, server-side. Never send it to the frontend.
# NOTE: "Groq" (console.groq.com, fast inference) is a different company from "Grok" (xAI).
# playai-tts was deprecated 12/31/25 -- replaced by canopylabs/orpheus-v1-english.
# Orpheus voices: autumn, diana, hannah (female) / austin, daniel, troy (male).
# IMPORTANT: Orpheus input is capped at 200 characters -- longer text is rejected.
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_TTS_URL = "https://api.groq.com/openai/v1/audio/speech"
GROQ_TTS_MODEL = "canopylabs/orpheus-v1-english"
DEFAULT_VOICE = "autumn"
ORPHEUS_CHAR_LIMIT = 200

app = FastAPI(title="MediCore AI Backend", version="1.0.0")

# CORS — tighten allow_origins to your actual frontend domain before production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _process(file_bytes: bytes):
    """Shared pipeline: load -> clean -> compute metrics. Returns everything downstream needs."""
    raw_df, df, cm, val_errors_raw, cleaning_log, attrs = core.load_and_prepare(file_bytes)
    quality = core.compute_quality(df)
    hs = core.compute_hospital_score(df, quality["quality_score"], cm)
    maturity = core.compute_maturity(quality, hs)
    return raw_df, df, cm, val_errors_raw, cleaning_log, attrs, quality, hs, maturity


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    """Full analytics snapshot: KPIs, schema coverage, data quality, clinical domains."""
    try:
        content = await file.read()
        raw_df, df, cm, val_errors_raw, cleaning_log, attrs, quality, hs, maturity = _process(content)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    schema_fields = ["patient_id","age","gender","condition","procedure","cost",
                      "length_of_stay","readmission","outcome","satisfaction",
                      "admission_dt","discharge_dt","ward","doctor","insurance"]

    return {
        "patients": len(df),
        "size_category": attrs["size_cat"],
        "schema_coverage": {f: (cm.get(f) is not None) for f in schema_fields},
        "clinical_domains": list(attrs["clinical_domains"].keys()),
        "data_quality": quality,
        "hospital_score": hs,
        "maturity": maturity,
        "validation_warnings": val_errors_raw,
        "cleaning_log": cleaning_log,
    }


@app.post("/api/findings")
async def get_findings(file: UploadFile = File(...), top_n: int = 20):
    """
    Returns the findings[] contract for the step-confirm frontend:
    ordered list of {id, severity, kpi, chart, explanation}.
    """
    try:
        content = await file.read()
        raw_df, df, cm, val_errors_raw, cleaning_log, attrs, quality, hs, maturity = _process(content)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    result = findings_module.build_findings(
        df, cm, hs, quality, maturity, attrs["clinical_domains"], top_n=top_n
    )
    return {"findings": result, "patients": len(df)}


@app.post("/api/report/pdf")
async def report_pdf(
    file: UploadFile = File(...),
    hospital_name: str = Form(""),
    report_period: str = Form(""),
):
    try:
        content = await file.read()
        raw_df, df, cm, val_errors_raw, cleaning_log, attrs, quality, hs, maturity = _process(content)
        pdf_bytes = reports.generate_pdf_report(df, cm, quality, hs, maturity, hospital_name, report_period)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ImportError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=medicore_ai_report.pdf"},
    )


@app.post("/api/report/pptx")
async def report_pptx(file: UploadFile = File(...)):
    try:
        content = await file.read()
        raw_df, df, cm, val_errors_raw, cleaning_log, attrs, quality, hs, maturity = _process(content)
        pptx_bytes = reports.generate_pptx_report(df, cm, quality, hs, maturity)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ImportError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return StreamingResponse(
        io.BytesIO(pptx_bytes),
        media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        headers={"Content-Disposition": "attachment; filename=medicore_ai_presentation.pptx"},
    )


def _truncate_for_orpheus(text: str, limit: int = ORPHEUS_CHAR_LIMIT) -> str:
    """Orpheus rejects input over 200 chars. Cut at the last sentence boundary
    (period/!/?) before the limit; fall back to the last space; never cut mid-word
    unless the text has no spaces at all."""
    if len(text) <= limit:
        return text
    window = text[:limit]
    for punct in (". ", "! ", "? "):
        idx = window.rfind(punct)
        if idx != -1:
            return window[: idx + 1].strip()
    idx = window.rfind(" ")
    if idx != -1:
        return window[:idx].strip()
    return window.strip()


@app.post("/api/tts")
async def text_to_speech(text: str = Form(...), voice_id: str = Form(DEFAULT_VOICE)):
    """
    Proxies to Groq's Orpheus TTS API (api.groq.com/openai/v1/audio/speech).
    Frontend sends plain text (a finding's explanation), gets back WAV bytes.
    GROQ_API_KEY never leaves this server. Text over 200 chars is truncated at a
    sentence boundary -- Orpheus hard-rejects anything longer.
    """
    if not GROQ_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY not set on the server. Add it in Render -> Environment.",
        )
    if not voice_id:
        voice_id = DEFAULT_VOICE
    text = _truncate_for_orpheus(text)

    try:
        resp = requests.post(
            GROQ_TTS_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": GROQ_TTS_MODEL,
                "input": text,
                "voice": voice_id,
                "response_format": "wav",
            },
            timeout=30,
        )
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Could not reach Groq TTS: {e}")

    if resp.status_code != 200:
        raise HTTPException(
            status_code=resp.status_code,
            detail=f"Groq TTS error: {resp.text[:300]}",
        )

    return StreamingResponse(io.BytesIO(resp.content), media_type="audio/wav")
