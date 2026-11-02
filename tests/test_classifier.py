"""Testes do classificador de sentimento pronto para uso (treinar, prever, salvar, carregar)."""

from pathlib import Path

import pytest

from delivery_ai.classifier import SentimentClassifier
from delivery_ai.config import NEGATIVE, POSITIVE, SENTIMENT_NAMES


def test_predict_returns_sentiment_label_and_probability(
    trained_classifier: SentimentClassifier,
) -> None:
    """Cada previsão traz o texto, o sentimento (número e nome) e a chance de ser negativo."""
    happy, unhappy = trained_classifier.predict(["adorei", "pessimo"])

    assert happy.text == "adorei"
    assert (happy.sentiment, happy.label) == (POSITIVE, SENTIMENT_NAMES[POSITIVE])
    assert (unhappy.sentiment, unhappy.label) == (NEGATIVE, SENTIMENT_NAMES[NEGATIVE])
    assert 0.0 <= happy.negative_probability < unhappy.negative_probability <= 1.0


def test_predict_without_texts_returns_empty_list(trained_classifier: SentimentClassifier) -> None:
    """Uma lista vazia não deve quebrar quem chama (o scikit-learn rejeitaria zero linhas)."""
    assert trained_classifier.predict([]) == []


def test_save_and_load_keep_the_same_predictions(
    trained_classifier: SentimentClassifier, tmp_path: Path
) -> None:
    """O modelo carregado do disco responde igual ao original, e a pasta é criada se faltar."""
    model_path = tmp_path / "models" / "sentiment_model.joblib"
    texts = ["adorei", "pessimo", "otimo produto"]

    saved_path = trained_classifier.save(model_path)
    loaded = SentimentClassifier.load(model_path)

    assert saved_path == model_path
    assert loaded.predict(texts) == trained_classifier.predict(texts)


def test_load_missing_model_explains_how_to_train(tmp_path: Path) -> None:
    """Sem o arquivo, o erro diz o que fazer em vez de só reclamar."""
    with pytest.raises(FileNotFoundError, match="python -m delivery_ai.train"):
        SentimentClassifier.load(tmp_path / "inexistente.joblib")
