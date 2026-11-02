"""Construtores dos modelos de sentimento da v1.

Cada função devolve um modelo novo, ainda não treinado, para que notebook e testes
criem os modelos sempre com a mesma configuração.
"""

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from delivery_ai.config import TFIDF_MIN_DF, UNIGRAMS

# Nomes das etapas do pipeline, para acessar o vocabulário e os pesos depois do treino.
VECTORIZER_STEP = "tfidf"
CLASSIFIER_STEP = "classifier"


def build_baseline_model() -> DummyClassifier:
    """Modelo "burro" que sempre prevê a classe mais comum do treino.

    Serve de piso: um modelo de verdade só vale a pena se superar este com folga.
    """
    return DummyClassifier(strategy="most_frequent")


def build_tfidf_vectorizer(ngram_range: tuple[int, int] = UNIGRAMS) -> TfidfVectorizer:
    """Transforma textos em vetores TF-IDF, com o vocabulário aprendido no `fit`.

    Acentos são removidos porque os clientes escrevem "otimo" e "ótimo" com o mesmo sentido.
    Não há lista de stopwords de propósito: as listas comuns de português incluem "não",
    que inverte o sentimento da frase. Com bigramas, pares como "nao recebi" ganham coluna
    própria e a negação deixa de ser uma palavra solta.
    """
    return TfidfVectorizer(strip_accents="unicode", min_df=TFIDF_MIN_DF, ngram_range=ngram_range)


def build_sentiment_model(
    ngram_range: tuple[int, int] = UNIGRAMS,
    class_weight: str | None = None,
) -> Pipeline:
    """Modelo da v1: TF-IDF seguido de regressão logística, num único pipeline.

    Juntar as duas etapas garante que o vocabulário seja aprendido só nos dados do `fit`
    (sem vazamento) e deixa um único objeto para salvar e usar na API com texto cru.
    `class_weight="balanced"` dá mais peso aos erros na classe minoritária (negativa),
    trocando precisão por recall.
    """
    return Pipeline(
        [
            (VECTORIZER_STEP, build_tfidf_vectorizer(ngram_range)),
            (CLASSIFIER_STEP, LogisticRegression(class_weight=class_weight)),
        ]
    )
