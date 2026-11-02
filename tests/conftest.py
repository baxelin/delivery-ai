"""Dados e modelos pequenos compartilhados por vários arquivos de teste."""

import pytest

from delivery_ai.classifier import SentimentClassifier
from delivery_ai.config import NEGATIVE, POSITIVE


@pytest.fixture
def separable_reviews() -> tuple[list[str], list[int]]:
    """Frases em que uma palavra decide a classe: qualquer modelo razoável acerta tudo."""
    texts = [
        "adorei produto",
        "adorei entrega",
        "otimo produto",
        "otimo entrega",
        "pessimo produto",
        "pessimo entrega",
        "horrivel produto",
        "horrivel entrega",
    ]
    labels = [POSITIVE] * 4 + [NEGATIVE] * 4
    return texts, labels


@pytest.fixture
def trained_classifier(separable_reviews: tuple[list[str], list[int]]) -> SentimentClassifier:
    """Classificador da v1 treinado nas frases separáveis."""
    texts, labels = separable_reviews
    return SentimentClassifier.train(texts, labels)
