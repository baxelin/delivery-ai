"""Testes das métricas usadas para comparar os classificadores de sentimento."""

import pytest
from sklearn.exceptions import NotFittedError

from delivery_ai.config import NEGATIVE, POSITIVE
from delivery_ai.evaluation import cross_validate_predictions, evaluate_predictions
from delivery_ai.models import build_sentiment_model


def test_evaluate_predictions_perfect_predictions_score_one() -> None:
    """Acertar tudo deve dar nota máxima em todas as métricas."""
    labels = [NEGATIVE, NEGATIVE, POSITIVE, POSITIVE]

    metrics = evaluate_predictions(labels, labels)

    assert metrics.accuracy == 1.0
    assert metrics.negative_precision == 1.0
    assert metrics.negative_recall == 1.0
    assert metrics.negative_f1 == 1.0
    assert metrics.macro_f1 == 1.0


@pytest.mark.filterwarnings("error")
def test_evaluate_predictions_always_positive_has_zero_negative_f1() -> None:
    """O caso do baseline: acurácia alta, mas nenhum negativo encontrado.

    Sem warnings, porque a divisão por zero (nenhum negativo previsto) é tratada de forma explícita.
    """
    y_true = [POSITIVE, POSITIVE, POSITIVE, NEGATIVE]
    y_pred = [POSITIVE, POSITIVE, POSITIVE, POSITIVE]

    metrics = evaluate_predictions(y_true, y_pred)

    assert metrics.accuracy == pytest.approx(0.75)
    assert metrics.negative_precision == 0.0
    assert metrics.negative_recall == 0.0
    assert metrics.negative_f1 == 0.0


def test_evaluate_predictions_mixed_case_matches_hand_calculation() -> None:
    """Valores conferidos à mão: 1 dos 2 negativos encontrado, sem alarme falso."""
    y_true = [NEGATIVE, NEGATIVE, POSITIVE, POSITIVE]
    y_pred = [NEGATIVE, POSITIVE, POSITIVE, POSITIVE]

    metrics = evaluate_predictions(y_true, y_pred)

    assert metrics.accuracy == pytest.approx(0.75)
    assert metrics.negative_precision == pytest.approx(1.0)
    assert metrics.negative_recall == pytest.approx(0.5)
    assert metrics.negative_f1 == pytest.approx(2 / 3)
    assert metrics.macro_f1 == pytest.approx((2 / 3 + 0.8) / 2)


def make_separable_reviews() -> tuple[list[str], list[int]]:
    """Avaliações em que uma palavra decide a classe: dá para acertar tudo."""
    positive_texts = [f"adorei pedido {index}" for index in range(10)]
    negative_texts = [f"pessimo pedido {index}" for index in range(10)]
    return positive_texts + negative_texts, [POSITIVE] * 10 + [NEGATIVE] * 10


def test_cross_validate_predictions_scores_separable_data_perfectly() -> None:
    """Cada avaliação é prevista por um modelo que não a viu, e ainda assim tudo é acertado."""
    texts, labels = make_separable_reviews()

    metrics = cross_validate_predictions(build_sentiment_model(), texts, labels)

    assert metrics.accuracy == 1.0
    assert metrics.negative_f1 == 1.0


def test_cross_validate_predictions_does_not_train_the_given_model() -> None:
    """A validação treina cópias; o modelo original segue sem treino para o uso final."""
    texts, labels = make_separable_reviews()
    model = build_sentiment_model()

    cross_validate_predictions(model, texts, labels)

    with pytest.raises(NotFittedError):
        model.predict(["adorei"])
