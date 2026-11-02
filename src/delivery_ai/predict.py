"""Classifica o sentimento de frases pelo terminal, usando o modelo salvo.

Uso: python -m delivery_ai.predict "chegou atrasado e amassado" "adorei, chegou rápido"

É o primeiro passo para a API da v5: um endpoint vai chamar estas mesmas funções.
"""

import sys
from collections.abc import Sequence

from delivery_ai.classifier import SentimentClassifier, SentimentPrediction

# Mostrados quando o script roda sem frases, para dar um exemplo de uso imediato.
EXAMPLE_REVIEWS = (
    "chegou atrasado e amassado",
    "adorei, chegou antes do prazo",
    "não tive nenhum problema, recomendo",
)


def predict_sentiment(
    text: str, classifier: SentimentClassifier | None = None
) -> SentimentPrediction:
    """Classifica uma frase.

    Sem classificador, carrega o modelo do disco a cada chamada; para várias frases,
    carregue uma vez e passe o mesmo classificador.
    """
    classifier = classifier or SentimentClassifier.load()
    return classifier.predict([text])[0]


def format_prediction(prediction: SentimentPrediction) -> str:
    """Transforma a previsão numa linha legível para o terminal."""
    return (
        f'"{prediction.text}" -> {prediction.label} '
        f"({prediction.negative_probability:.0%} de chance de ser negativo)"
    )


def main(texts: Sequence[str] | None = None, classifier: SentimentClassifier | None = None) -> None:
    """Classifica as frases recebidas pela linha de comando (ou exemplos, se não houver)."""
    texts = list(sys.argv[1:] if texts is None else texts) or list(EXAMPLE_REVIEWS)
    classifier = classifier or SentimentClassifier.load()
    for prediction in classifier.predict(texts):
        print(format_prediction(prediction))


if __name__ == "__main__":
    main()
