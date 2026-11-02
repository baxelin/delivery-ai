"""Construtores dos modelos de sentimento da v1.

Cada função devolve um modelo novo, ainda não treinado, para que notebook e testes
criem os modelos sempre com a mesma configuração.
"""

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from delivery_ai.config import TFIDF_MIN_DF

# Nomes das etapas do pipeline, para acessar o vocabulário e os pesos depois do treino.
VECTORIZER_STEP = "tfidf"
CLASSIFIER_STEP = "classifier"


def build_baseline_model() -> DummyClassifier:
    """Modelo "burro" que sempre prevê a classe mais comum do treino.

    Serve de piso: um modelo de verdade só vale a pena se superar este com folga.
    """
    return DummyClassifier(strategy="most_frequent")


def build_tfidf_vectorizer() -> TfidfVectorizer:
    """Transforma textos em vetores TF-IDF, com o vocabulário aprendido no `fit`.

    Acentos são removidos porque os clientes escrevem "otimo" e "ótimo" com o mesmo sentido.
    Não há lista de stopwords de propósito: as listas comuns de português incluem "não",
    que inverte o sentimento da frase.
    """
    return TfidfVectorizer(strip_accents="unicode", min_df=TFIDF_MIN_DF)


def build_sentiment_model() -> Pipeline:
    """Modelo da v1: TF-IDF seguido de regressão logística, num único pipeline.

    Juntar as duas etapas garante que o vocabulário seja aprendido só nos dados do `fit`
    (sem vazamento) e deixa um único objeto para salvar e usar na API com texto cru.
    """
    return Pipeline(
        [
            (VECTORIZER_STEP, build_tfidf_vectorizer()),
            (CLASSIFIER_STEP, LogisticRegression()),
        ]
    )
