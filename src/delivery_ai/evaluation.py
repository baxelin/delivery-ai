"""Métricas para comparar classificadores de sentimento sempre do mesmo jeito.

O foco é a classe negativa: achar cliente insatisfeito é o que interessa ao negócio,
e com 71% de positivas a acurácia sozinha esconde um modelo que nunca acha nenhum.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from delivery_ai.config import CV_FOLDS, NEGATIVE, RANDOM_STATE

# Quando o modelo não prevê nenhum negativo, precisão e F1 viram 0/0; a regra é tratar como 0.
ZERO_DIVISION_VALUE = 0.0


@dataclass(frozen=True)
class ClassificationMetrics:
    """Resultado da avaliação de um classificador, com as métricas da classe negativa."""

    accuracy: float
    negative_precision: float
    negative_recall: float
    negative_f1: float
    macro_f1: float


def evaluate_predictions(y_true: Sequence[int], y_pred: Sequence[int]) -> ClassificationMetrics:
    """Calcula as métricas de um conjunto de previsões contra os rótulos verdadeiros."""
    negative_class_options = {"pos_label": NEGATIVE, "zero_division": ZERO_DIVISION_VALUE}
    return ClassificationMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        negative_precision=float(precision_score(y_true, y_pred, **negative_class_options)),
        negative_recall=float(recall_score(y_true, y_pred, **negative_class_options)),
        negative_f1=float(f1_score(y_true, y_pred, **negative_class_options)),
        macro_f1=float(
            f1_score(y_true, y_pred, average="macro", zero_division=ZERO_DIVISION_VALUE)
        ),
    )


def cross_validate_predictions(
    model: BaseEstimator,
    texts: Sequence[str],
    labels: Sequence[int],
    folds: int = CV_FOLDS,
    random_state: int = RANDOM_STATE,
) -> ClassificationMetrics:
    """Avalia uma configuração só com o treino, por validação cruzada estratificada.

    Serve para comparar configurações sem olhar o teste: escolher olhando o teste
    tornaria a nota final otimista. O modelo recebido não é treinado (o scikit-learn
    treina cópias), e as divisões repetem com a mesma semente para a comparação ser justa.
    """
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    predictions = cross_val_predict(model, texts, labels, cv=splitter)
    return evaluate_predictions(labels, predictions)
