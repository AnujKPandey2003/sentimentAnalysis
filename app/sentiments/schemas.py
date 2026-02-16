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