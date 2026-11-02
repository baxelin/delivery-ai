"""Testes dos construtores de modelos de sentimento."""

import pandas as pd

from delivery_ai.config import NEGATIVE, POSITIVE
from delivery_ai.models import build_baseline_model


def test_baseline_model_always_predicts_majority_class() -> None:
    """O baseline ignora o texto: prevê a classe mais comum do treino para qualquer entrada."""
    train_texts = pd.Series(["ótimo", "chegou rápido", "adorei", "péssimo"])
    train_labels = pd.Series([POSITIVE, POSITIVE, POSITIVE, NEGATIVE])
    unseen_texts = pd.Series(["produto quebrado, quero reembolso", "nunca chegou"])

    model = build_baseline_model().fit(train_texts, train_labels)

    assert model.predict(unseen_texts).tolist() == [POSITIVE, POSITIVE]
