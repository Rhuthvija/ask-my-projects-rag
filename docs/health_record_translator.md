# AI Health Record Translator

Upload medical records in **any mix of formats** (PDF, scanned PDF, JPG/PNG/HEIC photos, DOCX/DOC, TXT, CSV, XLSX, JSON) and get **one plain-language health summary**: abnormal values, trends, doctor-question prompts, and an "Ask My Report" chatbot that answers only from your own records. English / Hindi / Telugu.

It is an **informational tool**. It never diagnoses, and never gives medicine or dosage advice.

## Run it

Requirements: Python 3.10+, Node 18+, and the Tesseract OCR engine (`sudo apt install tesseract-ocr` / `brew install tesseract`).

```bash
./run.sh            # builds the UI, installs Python deps, serves http://localhost:8000
```

Then sign up, open **Upload**, and press **Load demo data** (7 fictional records in 7 formats).

Dev mode with hot reload: `cd backend && uvicorn app.main:app --reload` and, in another terminal, `cd frontend && npm run dev` (http://localhost:5173, proxies /api to :8000).

Tests: `cd backend && python -m pytest tests -q`

## How it works (the "model")

This is a **pipeline, not one trained network** - on purpose. Only the wording task is language-like; the rest must be exact and checkable:

| Step | What runs | Why |
|---|---|---|
| File type detection | magic bytes + extension | users rename files |
| Text extraction | pdfplumber (text layer), python-docx, pandas/openpyxl (CSV/XLSX/JSON), Tesseract OCR for photos and scanned PDF pages | each format has its own best extractor |
| Value extraction | `pipeline/parser.py` (name / value / unit / printed range, dates, patient sex, doc type) and `structured.py` (long + wide tables) | deterministic, testable |
| Normalization | `knowledge/parameters.py`: synonyms (Hb/HGB/Hemoglobin), unit conversion (mmol/L, lakhs/cumm, /cumm, °F ...) | one common structure |
| Abnormality detection | `pipeline/analyze.py`: inside range = normal; outside by <=10% of the range width = "needs attention"; more = outside | transparent rule, no black box |
| Explanations | `ai/explain.py` templates (EN/HI/TE); optional Claude API for the summary and chatbot | cannot hallucinate; LLM optional |
| Ask My Report | `ai/chat.py` intent routing over your extracted data; `ai/safety.py` blocks diagnosis / dosage wording | grounded + safe |

Every value carries: test, value, unit, reference range (and whether it came from *your report* or a *general adult range*), date, status, **source file + page**, confidence, and a "please verify" flag. You can correct any value; the status is recomputed.

### Safety rules built in
* A range printed in your report always wins. A general adult range is only a labelled fallback.
* Never guessed: sex-specific ranges (hemoglobin, creatinine, HDL ...) without a known sex, and random glucose, show "Unable to assess".
* OCR text under 75% confidence, a missing unit, or an implausible value is flagged "Please verify" - never silently trusted.
* The chatbot answers "I couldn't find that information in the uploaded records." rather than guessing; questions like "Do I have diabetes?" or "what dose should I take?" get a safe redirect.
* The "score" on the dashboard is just the share of assessable values inside range - it is not a medical risk score.

### Privacy & security implemented
Email + scrypt-hashed password, JWT sessions, login throttling, every query scoped to the user, uploaded files encrypted at rest (Fernet), `Cache-Control: no-store`, delete one report / all data / the account, JSON export. **Serve over HTTPS in any real deployment** (put it behind a TLS reverse proxy).

## Optional AI (LLM) mode
```bash
export ANTHROPIC_API_KEY=sk-ant-...      # optional
export ANTHROPIC_MODEL=claude-sonnet-4-5 # optional, any model your key can use
```
With a key, Ask My Report answers in the selected language and the dashboard gets a friendlier summary. Only extracted values are sent (never the files), and every answer passes the safety guard, falling back to offline mode if it fails. Without a key everything still works offline.

## Extending
* **New test**: add one entry to `PARAMS` in `backend/app/knowledge/parameters.py` (aliases, unit conversions, range, description in EN/HI/TE).
* **New language**: add a block in `ai/i18n.py` and descriptions in `PARAMS`.
* **PostgreSQL**: re-implement the ~6 helpers in `db.py`.
* **Hindi/Telugu-language source reports**: `sudo apt install tesseract-ocr-hin tesseract-ocr-tel` and set `HRT_OCR_LANGS=eng+hin+tel` (parser aliases are English only for now).

## Known limitations (please read before presenting or deploying)
* Reference ranges in `parameters.py` are generic adult values for a prototype; they have **not** been clinically validated. Labs differ - that is why report-printed ranges take priority.
* Hindi/Telugu text is template-based and should be reviewed by a native speaker (and ideally a clinician) before real use.
* Legacy `.doc` files are converted through LibreOffice (must be installed); this was tested for text only, so if tables from a Word-authored `.doc` come out flattened, save the file as .docx or PDF.
* **Tested on layouts modelled on real lab reports** (Drlogy CBC, LFT, KFT, thyroid, lipid; see `backend/demo/real_layouts/` and `tests/test_real_layouts.py`): all 50 values are read correctly from clean PDFs. Those PDFs are *reconstructions* of the public layouts (fictional values), not the original files. On deliberately degraded scans some digits are misread by OCR; the app flags every such case "please verify" (low OCR confidence, stray symbols, or a mismatch with the High/Low label the report itself prints), but it cannot guarantee it catches every OCR error. Always compare with the original.
* 53 tests are recognised (CBC incl. differential/ESR, lipids, liver, kidney, electrolytes, thyroid, diabetes, vitals). Anything else on a report is skipped rather than guessed; add it to `PARAMS`.
* Photos of handwritten prescriptions are not supported; printed text only. Medication names/doses are deliberately not extracted.
* No email verification or password reset yet; SQLite is single-node.
* Demo records are fictional and clearly labelled as such.

## Layout
```
backend/app/  main.py (API)  auth.py  storage.py  db.py  analysis.py
  pipeline/   extractors  parser  structured  normalize  analyze  orchestrate
  knowledge/  parameters (ranges, synonyms, units)  glossary (abbreviations + jargon)
  ai/         explain  chat  llm  safety  i18n
backend/demo/generate_demo.py   backend/demo/real_layouts/   backend/tests/test_e2e.py  test_real_layouts.py
frontend/src/ pages/*  components/*  api.js  ctx.jsx
```
