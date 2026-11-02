"""Treina o classificador de sentimento da v1 e salva em `models/`.

Uso: python -m delivery_ai.train

Refaz o caminho do notebook (carregar, limpar, dividir, treinar, avaliar) num script
reproduzível, já que a pasta `models/` não vai para o git.
"""

from dataclasses import asdict
from pathlib import Path

from delivery_ai.classifier import SentimentClassifier
from delivery_ai.config import MESSAGE_COLUMN, REVIEWS_PATH, SENTIMENT_COLUMN, SENTIMENT_MODEL_PATH
from delivery_ai.data import load_reviews, prepare_sentiment_dataset, split_train_test
from delivery_ai.evaluation import ClassificationMetrics, evaluate_predictions


def train_and_save(
    reviews_path: Path = REVIEWS_PATH,
    model_path: Path = SENTIMENT_MODEL_PATH,
) -> ClassificationMetrics:
    """Treina, avalia no teste e salva o modelo, devolvendo as métricas do teste.

    Treina só com a parte de treino, para que as métricas impressas valham
    exatamente para o modelo que fica salvo.
    """
    split = split_train_test(prepare_sentiment_dataset(load_reviews(reviews_path)))
    classifier = SentimentClassifier.train(
        split.train[MESSAGE_COLUMN], split.train[SENTIMENT_COLUMN]
    )
    predictions = classifier.predict(split.test[MESSAGE_COLUMN])
    metrics = evaluate_predictions(
        split.test[SENTIMENT_COLUMN], [prediction.sentiment for prediction in predictions]
    )
    classifier.save(model_path)
    return metrics


def main() -> None:
    """Treina com os dados reais e mostra as métricas no terminal."""
    metrics = train_and_save()
    print(f"Modelo salvo em {SENTIMENT_MODEL_PATH}")
    print("Métricas no teste:")
    for name, value in asdict(metrics).items():
        print(f"  {name}: {value:.3f}")


if __name__ == "__main__":
    main()
