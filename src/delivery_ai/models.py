"""Construtores dos modelos de sentimento da v1.

Cada função devolve um modelo novo, ainda não treinado, para que notebook e testes
criem os modelos sempre com a mesma configuração.
"""

from sklearn.dummy import DummyClassifier


def build_baseline_model() -> DummyClassifier:
    """Modelo "burro" que sempre prevê a classe mais comum do treino.

    Serve de piso: um modelo de verdade só vale a pena se superar este com folga.
    """
    return DummyClassifier(strategy="most_frequent")
