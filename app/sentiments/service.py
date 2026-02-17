from .model import load_model


def analyze_text(text: str):
    classifier = load_model()

    result = classifier(text)[0]

    label = result["label"].lower()
    score = float(result["score"])
    confidence = round(score * 100, 2)

    return {
        "label": label,
        "score": score,
        "confidence": confidence
    }


def analyze_batch(texts: list):
    classifier = load_model()

    results = classifier(texts)

    formatted_results = []
    sentiment_sum = 0.0

    for r in results:
        label = r["label"].lower()
        score = float(r["score"])

        # Convert positive to +score, negative to -score
        if label == "positive":
            sentiment_sum += score
        elif label == "negative":
            sentiment_sum -= score

        formatted_results.append({
            "label": label,
            "score": score,
            "confidence": round(score * 100, 2)
        })

    avg_score = sentiment_sum / len(results)

    # Determine overall sentiment
    if avg_score > 0.2:
        overall = "positive"
    elif avg_score < -0.2:
        overall = "negative"
    else:
        overall = "neutral"

    return {
        "overall_sentiment": overall,
        "average_score": round(avg_score, 4),
        "confidence": round(abs(avg_score) * 100, 2),
        "results": formatted_results
    }