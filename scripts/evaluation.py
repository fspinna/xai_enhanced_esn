from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    jaccard_score,
)


CLASSIFICATION_PREDICT_METRICS = {
    "accuracy": accuracy_score,
    "balanced_accuracy": balanced_accuracy_score,
    "precision": precision_score,
    "recall": recall_score,
    "jaccard": jaccard_score,
    "f1": f1_score,
}

CLASSIFICATION_PREDICT_METRICS_KWARGS = {
    "accuracy": dict(),
    "balanced_accuracy": dict(),
    "precision": {"average": "macro"},
    "recall": {"average": "macro"},
    "jaccard": {"average": "macro"},
    "f1": {"average": "macro"},
}


def score_classification_predict(y_true, y_pred):
    scores = dict()
    for name, metric in CLASSIFICATION_PREDICT_METRICS.items():
        scores[name] = metric(
            y_true, y_pred, **CLASSIFICATION_PREDICT_METRICS_KWARGS[name]
        )
    return scores
