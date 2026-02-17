from pydantic import BaseModel
from typing import List


class SentimentRequest(BaseModel):
    text: str


class BatchSentimentRequest(BaseModel):
    texts: List[str]


class SentimentResponse(BaseModel):
    label: str
    score: float
    confidence: float


class BatchSentimentResponse(BaseModel):
    overall_sentiment: str
    average_score: float
    confidence: float
    results: List[SentimentResponse]