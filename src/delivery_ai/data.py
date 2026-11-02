"""Carregamento, limpeza e rotulagem das avaliações da Olist para o classificador de sentimento.

São funções puras: recebem um DataFrame e devolvem outro, sem alterar o original.
"""

from pathlib import Path

import pandas as pd

from delivery_ai.config import (
    ID_COLUMN,
    MESSAGE_COLUMN,
    NEUTRAL_SCORE,
    REVIEWS_PATH,
    SCORE_COLUMN,
    SCORE_TO_SENTIMENT,
    SENTIMENT_COLUMN,
)

# Só o que o modelo e a análise de erros precisam; o resto do CSV não é usado.
DATASET_COLUMNS = [ID_COLUMN, MESSAGE_COLUMN, SCORE_COLUMN, SENTIMENT_COLUMN]


def load_reviews(path: Path = REVIEWS_PATH) -> pd.DataFrame:
    """Lê o CSV de avaliações; o caminho padrão vem da config para todos usarem o mesmo arquivo."""
    return pd.read_csv(path)


def keep_reviews_with_text(reviews: pd.DataFrame) -> pd.DataFrame:
    """Mantém só avaliações com comentário, pois sem texto não há o que classificar."""
    has_text = reviews[MESSAGE_COLUMN].fillna("").str.strip() != ""
    return reviews[has_text]


def drop_duplicate_reviews(reviews: pd.DataFrame) -> pd.DataFrame:
    """Remove repetições de `review_id`: a mesma avaliação ligada a pedidos diferentes."""
    return reviews.drop_duplicates(subset=ID_COLUMN, keep="first")


def drop_neutral_reviews(reviews: pd.DataFrame) -> pd.DataFrame:
    """Descarta a nota neutra, que não é claramente positiva nem negativa."""
    return reviews[reviews[SCORE_COLUMN] != NEUTRAL_SCORE]


def add_sentiment_label(reviews: pd.DataFrame) -> pd.DataFrame:
    """Cria a coluna de sentimento a partir da nota.

    Falha se houver nota sem rótulo definido (como a 3), para que um erro na ordem
    da limpeza não vire rótulo errado em silêncio.
    """
    labels = reviews[SCORE_COLUMN].map(SCORE_TO_SENTIMENT)
    invalid_scores = sorted(
        int(score) for score in reviews.loc[labels.isna(), SCORE_COLUMN].unique()
    )
    if invalid_scores:
        raise ValueError(
            f"Notas sem rótulo de sentimento: {invalid_scores}. "
            f"Remova a nota {NEUTRAL_SCORE} antes de rotular."
        )
    return reviews.assign(**{SENTIMENT_COLUMN: labels.astype(int)})


def prepare_sentiment_dataset(reviews: pd.DataFrame) -> pd.DataFrame:
    """Aplica toda a limpeza e a rotulagem, devolvendo o conjunto pronto para treino."""
    prepared = (
        reviews.pipe(keep_reviews_with_text)
        .pipe(drop_duplicate_reviews)
        .pipe(drop_neutral_reviews)
        .pipe(add_sentiment_label)
    )
    return prepared[DATASET_COLUMNS].reset_index(drop=True)
