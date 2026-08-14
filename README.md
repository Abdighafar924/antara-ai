# MediCore AI Backend

Stateless FastAPI service that ports the original MediCore Streamlit analytics engine
(adaptive column detection, cleaning, clinical domain panels, scoring, PDF/PPTX reports)
into an API a React frontend can call.

## Structure
```
medicore_backend/
  app/
    core.py       # data loading, cleaning, quality/hospital-score/maturity, clustering
    findings.py   # turns metrics into the findings[] contract for the step-confirm UI
    reports.py    # PDF (ReportLab) + PPTX (python-pptx) generators
    main.py       # FastAPI endpoints
  requirements.txt
  render.yaml
```

## Run locally
```bash
cd medicore_backend
python -m venv .venv
.venv\Scripts\activate        # PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```
Docs at http://localhost:8000/docs (FastAPI auto-generates a test UI — upload a CSV there first).

## Endpoints
| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/health` | — | `{"status":"ok"}` |
| POST | `/api/analyze` | `file` (CSV) | KPIs, schema coverage, data quality, clinical domains |
| POST | `/api/findings` | `file` (CSV), `top_n` (query, default 8) | `findings[]` — the KPI→chart→explanation contract |
| POST | `/api/report/pdf` | `file`, `hospital_name`, `report_period` | PDF download |
| POST | `/api/report/pptx` | `file` | PPTX download |

## Deploy to Render
1. Push this folder to a GitHub repo (or a subfolder of an existing one — set Root Directory in Render if so).
2. Render dashboard → New → Blueprint → point at the repo → it reads `render.yaml` automatically.
   (Or: New → Web Service → Build Command `pip install -r requirements.txt` → Start Command
   `uvicorn app.main:app --host 0.0.0.0 --port $PORT`.)
3. Free tier spins down on idle — first request after sleep takes ~30-50s.

## Findings contract (what the React step-confirm UI consumes)
```json
{
  "findings": [
    {
      "id": "readmission",
      "severity": "critical",
      "kpi": {"label": "Readmission Rate", "value": "22.4%", "benchmark": "≤15%"},
      "chart": {"type": "bar", "labels": ["Diabetes","HTN"], "values": [28.1, 19.3], "y_label": "Readmission Rate (%)"},
      "explanation": "Readmission rate is 22.4%, above the 15% benchmark. Highest-risk condition: Diabetes (28.1%)."
    }
  ],
  "patients": 1204
}
```
Findings arrive pre-sorted: critical → warning → info → good. The frontend just steps through the array.

## Known scope cuts (v1)
- PDF report is a condensed ~7-section version, not the original ~20-page report — extend `reports.py` from the
  original Streamlit script's `generate_pdf_report` if you want full parity.
- PPTX is 4 slides (title, KPIs, clinical, recommendations) — same reasoning.
- No auth yet. Add an API-key header check in `main.py` before this touches real patient data
  (mirror how `clinic_api` does it in Clinical OS).
- No persistence — every request re-uploads and re-processes the CSV. Fine for MVP; add a
  `clinic_api` pull-based endpoint next so the frontend doesn't need manual CSV upload.
- Patient segmentation (KMeans) is ported in `core.run_clustering` but not yet wired to an endpoint.

## Next step
Point `/api/analyze` at your Clinical OS `clinic_api` export instead of a manual upload —
swap the `UploadFile` param for an internal `requests.get()` call to `clinic_api`, same
`core.load_and_prepare()` downstream.
# antara-ai
# antara-ai
