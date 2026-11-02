"""Testes do script que treina e salva o modelo da v1."""

from pathlib import Path

import pandas as pd

from delivery_ai.classifier import SentimentClassifier
from delivery_ai.config import ID_COLUMN, MESSAGE_COLUMN, NEGATIVE, SCORE_COLUMN, SENTIMENT_NAMES
from delivery_ai.train import train_and_save

REVIEWS_PER_CLASS = 20


def write_reviews_csv(path: Path) -> None:
    """CSV no formato da Olist, com casos que a limpeza precisa descartar."""
    rows = [(f"pos{i}", 5, f"adorei produto {i}") for i in range(REVIEWS_PER_CLASS)]
    rows += [(f"neg{i}", 1, f"pessimo produto {i}") for i in range(REVIEWS_PER_CLASS)]
    rows += [("neutra", 3, "ok"), ("sem_texto", 5, None)]
    pd.DataFrame(rows, columns=[ID_COLUMN, SCORE_COLUMN, MESSAGE_COLUMN]).to_csv(path, index=False)


def test_train_and_save_creates_model_and_reports_test_metrics(tmp_path: Path) -> None:
    """O script refaz o caminho do notebook e deixa o modelo pronto para o predict."""
    reviews_path = tmp_path / "reviews.csv"
    model_path = tmp_path / "models" / "sentiment_model.joblib"
    write_reviews_csv(reviews_path)

    metrics = train_and_save(reviews_path, model_path)

    assert model_path.exists()
    assert metrics.accuracy == 1.0
    prediction = SentimentClassifier.load(model_path).predict(["pessimo"])[0]
    assert prediction.label == SENTIMENT_NAMES[NEGATIVE]
