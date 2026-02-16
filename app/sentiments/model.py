from transformers import pipeline
from .config import MODEL_NAME
import logging

logger = logging.getLogger(__name__)

classifier = None


def load_model():
    global classifier

    if classifier is None:
        logger.info("Loading FinBERT model...")
        classifier = pipeline(
            "sentiment-analysis",
            model=MODEL_NAME
        )
        logger.info("Model loaded successfully.")

    return classifier