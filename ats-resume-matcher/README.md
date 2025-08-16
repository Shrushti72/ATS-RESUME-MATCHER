
# ATS Resume Matcher

## Setup
```bash
python -m venv .venv
# Windows
. .venv/Scripts/activate
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

## Run FastAPI backend
```bash
uvicorn app.Resume:app --host 0.0.0.0 --port 8000 --reload
```
- Health: `http://127.0.0.1:8000/health`
- Multipart: `POST /check_resume`
- JSON: `POST /check_resume_json`
- Header: `X-API-KEY: test123`

## Expose with ngrok
```bash
ngrok http 8000
```
Copy the `https://<id>.ngrok-free.app` and use it in the frontend field "API Base URL".

## Frontend
Open `frontend/index.html` in your browser. Enter your API Base URL and API Key, upload resume and paste JD.

## Streamlit UI (optional)
```bash
streamlit run streamlit_app.py
```
