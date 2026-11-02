"""Testes dos construtores de modelos de sentimento."""

import pandas as pd

from delivery_ai.config import NEGATIVE, POSITIVE
from delivery_ai.models import build_baseline_model, build_tfidf_vectorizer


def test_baseline_model_always_predicts_majority_class() -> None:
    """O baseline ignora o texto: prevê a classe mais comum do treino para qualquer entrada."""
    train_texts = pd.Series(["ótimo", "chegou rápido", "adorei", "péssimo"])
    train_labels = pd.Series([POSITIVE, POSITIVE, POSITIVE, NEGATIVE])
    unseen_texts = pd.Series(["produto quebrado, quero reembolso", "nunca chegou"])

    model = build_baseline_model().fit(train_texts, train_labels)

    assert model.predict(unseen_texts).tolist() == [POSITIVE, POSITIVE]


def test_tfidf_vectorizer_ignores_accents_and_case() -> None:
    """Acentos e maiúsculas não importam: "Ótimo" e "otimo" são a mesma palavra."""
    vectorizer = build_tfidf_vectorizer()

    matrix = vectorizer.fit_transform(["Ótimo produto", "otimo produto"])

    assert "otimo" in vectorizer.vocabulary_
    assert "ótimo" not in vectorizer.vocabulary_
    assert (matrix[0] != matrix[1]).nnz == 0


def test_tfidf_vectorizer_drops_words_seen_in_a_single_review() -> None:
    """Palavra que aparece numa avaliação só (como erro de digitação) não ajuda a generalizar."""
    vectorizer = build_tfidf_vectorizer().fit(
        ["entrega rapida", "entrega atrasada", "entrega rapdia"]
    )

    assert set(vectorizer.vocabulary_) == {"entrega"}


def test_tfidf_vectorizer_keeps_negation() -> None:
    """A palavra "não" inverte o sentido da frase e não pode ser descartada como comum."""
    vectorizer = build_tfidf_vectorizer().fit(["não chegou", "não gostei"])

    assert "nao" in vectorizer.vocabulary_


def test_tfidf_vectorizer_learns_vocabulary_only_from_fit_texts() -> None:
    """Palavras vistas só depois do fit (como as do teste) não ganham coluna: sem vazamento."""
    vectorizer = build_tfidf_vectorizer().fit(["produto bom", "produto bom demais"])

    unseen = vectorizer.transform(["péssimo atendimento"])

    assert "pessimo" not in vectorizer.vocabulary_
    assert unseen.shape[1] == len(vectorizer.vocabulary_)
    assert unseen.nnz == 0
