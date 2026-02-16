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

    for r in results:
        formatted_results.append({
            "label": r["label"].lower(),
            "score": float(r["score"]),
            "confidence": round(r["score"] * 100, 2)
        })

    return formatted_results