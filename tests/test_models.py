"""Testes dos construtores de modelos de sentimento."""

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from delivery_ai.config import NEGATIVE, POSITIVE, UNIGRAMS_AND_BIGRAMS
from delivery_ai.models import (
    CLASSIFIER_STEP,
    build_baseline_model,
    build_sentiment_model,
    build_tfidf_vectorizer,
)


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


@pytest.fixture
def trained_sentiment_model(separable_reviews: tuple[list[str], list[int]]) -> Pipeline:
    """Modelo treinado num conjunto pequeno em que as palavras separam bem as classes."""
    texts, labels = separable_reviews
    return build_sentiment_model().fit(texts, labels)


def test_sentiment_model_classifies_new_texts(trained_sentiment_model: Pipeline) -> None:
    """Textos que o modelo nunca viu, mas com palavras conhecidas, saem com o sentimento certo."""
    predictions = trained_sentiment_model.predict(["adorei", "horrivel demais"])

    assert predictions.tolist() == [POSITIVE, NEGATIVE]


def test_sentiment_model_uses_project_vectorizer(trained_sentiment_model: Pipeline) -> None:
    """Acento e maiúscula são tratados dentro do modelo, sem preparo extra de quem chama."""
    assert trained_sentiment_model.predict(["Péssimo"]).tolist() == [NEGATIVE]


def test_sentiment_model_returns_probabilities(trained_sentiment_model: Pipeline) -> None:
    """Uma probabilidade por classe, somando 1: base para medir a confiança de cada previsão."""
    probabilities = trained_sentiment_model.predict_proba(["adorei", "pessimo"])

    assert probabilities.shape == (2, 2)
    assert probabilities.sum(axis=1).tolist() == pytest.approx([1.0, 1.0])


def test_tfidf_vectorizer_with_bigrams_keeps_word_pairs() -> None:
    """Com bigramas, "nao recebi" vira uma coluna própria, separada de "nao tive"."""
    vectorizer = build_tfidf_vectorizer(ngram_range=UNIGRAMS_AND_BIGRAMS).fit(
        ["não recebi o pedido", "não recebi nada", "não tive problema", "não tive problemas"]
    )

    assert "nao recebi" in vectorizer.vocabulary_
    assert "nao tive" in vectorizer.vocabulary_


def test_tfidf_vectorizer_uses_only_single_words_by_default() -> None:
    """O padrão continua sendo o modelo inicial, para não mudar os resultados já registrados."""
    vectorizer = build_tfidf_vectorizer().fit(["não recebi", "não recebi"])

    assert set(vectorizer.vocabulary_) == {"nao", "recebi"}


def test_sentiment_model_passes_class_weight_to_classifier() -> None:
    """O peso das classes chega à regressão logística, que é quem o usa."""
    model = build_sentiment_model(class_weight="balanced")

    assert model.named_steps[CLASSIFIER_STEP].class_weight == "balanced"
