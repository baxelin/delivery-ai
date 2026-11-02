"""Testes do script de previsão pelo terminal."""

import pytest

from delivery_ai.classifier import SentimentClassifier, SentimentPrediction
from delivery_ai.config import NEGATIVE, POSITIVE, SENTIMENT_NAMES
from delivery_ai.predict import EXAMPLE_REVIEWS, format_prediction, main, predict_sentiment


def test_predict_sentiment_classifies_a_single_text(
    trained_classifier: SentimentClassifier,
) -> None:
    """A função de uso recebe uma frase e devolve uma previsão."""
    prediction = predict_sentiment("horrivel entrega", classifier=trained_classifier)

    assert prediction.label == SENTIMENT_NAMES[NEGATIVE]


def test_format_prediction_shows_text_label_and_probability() -> None:
    """A linha impressa precisa ser legível para quem não conhece o modelo."""
    prediction = SentimentPrediction(
        text="chegou atrasado", sentiment=NEGATIVE, negative_probability=0.97
    )

    assert format_prediction(prediction) == (
        '"chegou atrasado" -> negativo (97% de chance de ser negativo)'
    )


def test_main_prints_one_line_per_text(
    trained_classifier: SentimentClassifier, capsys: pytest.CaptureFixture[str]
) -> None:
    """Cada frase recebida vira uma linha com o seu sentimento."""
    main(["adorei", "pessimo"], classifier=trained_classifier)

    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 2
    assert lines[0].startswith(f'"adorei" -> {SENTIMENT_NAMES[POSITIVE]}')
    assert lines[1].startswith(f'"pessimo" -> {SENTIMENT_NAMES[NEGATIVE]}')


def test_main_without_texts_uses_examples(
    trained_classifier: SentimentClassifier, capsys: pytest.CaptureFixture[str]
) -> None:
    """Rodar sem argumentos mostra exemplos, em vez de não fazer nada."""
    main([], classifier=trained_classifier)

    assert len(capsys.readouterr().out.strip().splitlines()) == len(EXAMPLE_REVIEWS)
