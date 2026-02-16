import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .schemas import (
    SentimentRequest,
    BatchSentimentRequest,
    SentimentResponse
)

from .service import analyze_text, analyze_batch
from .utils import setup_logging

setup_logging()

app = FastAPI(
    title="Hedge Fund Sentiment Service",
    description="FinBERT-based Financial Sentiment Analysis API",
    version="1.0.0"
)

# CORS (for frontend + Spring Boot)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"status": "Sentiment API Running"}




@app.get("/health")
@app.head("/health")
def health():
    return {"status": "ok"}

@app.post("/analyze", response_model=SentimentResponse)
def analyze(request: SentimentRequest):
    try:
        return analyze_text(request.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze-batch")
def analyze_batch_endpoint(request: BatchSentimentRequest):
    try:
        return analyze_batch(request.texts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# IMPORTANT FOR RENDER
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.sentiments.main:app", host="0.0.0.0", port=port)