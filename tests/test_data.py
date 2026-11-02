"""Testes da limpeza, rotulagem e divisão treino/teste das avaliações da Olist."""

from pathlib import Path

import pandas as pd
import pytest

from delivery_ai.config import (
    ID_COLUMN,
    MESSAGE_COLUMN,
    NEGATIVE,
    POSITIVE,
    SCORE_COLUMN,
    SENTIMENT_COLUMN,
)
from delivery_ai.data import (
    add_sentiment_label,
    drop_duplicate_reviews,
    drop_neutral_reviews,
    keep_reviews_with_text,
    load_reviews,
    prepare_sentiment_dataset,
    split_train_test,
)


def make_reviews(rows: list[tuple[str, int, str | None]]) -> pd.DataFrame:
    """Monta um DataFrame mínimo, só com as colunas que a limpeza usa."""
    return pd.DataFrame(rows, columns=[ID_COLUMN, SCORE_COLUMN, MESSAGE_COLUMN])


def test_load_reviews_reads_csv(tmp_path: Path) -> None:
    """Garante que o carregamento lê o CSV sem perder linhas nem colunas."""
    csv_path = tmp_path / "reviews.csv"
    make_reviews([("a", 5, "Ótimo"), ("b", 1, None)]).to_csv(csv_path, index=False)

    result = load_reviews(csv_path)

    assert result[ID_COLUMN].tolist() == ["a", "b"]
    assert result[SCORE_COLUMN].tolist() == [5, 1]


def test_keep_reviews_with_text_removes_null_empty_and_blank_messages() -> None:
    """Mensagem só com espaços também não tem conteúdo para o modelo aprender."""
    reviews = make_reviews([("a", 5, "Ótimo"), ("b", 4, None), ("c", 1, ""), ("d", 2, "   ")])

    result = keep_reviews_with_text(reviews)

    assert result[ID_COLUMN].tolist() == ["a"]


def test_drop_duplicate_reviews_keeps_first_occurrence() -> None:
    """A mesma avaliação ligada a vários pedidos não pode contar duas vezes."""
    reviews = make_reviews([("a", 5, "Ótimo"), ("a", 5, "Ótimo"), ("b", 1, "Ruim")])

    result = drop_duplicate_reviews(reviews)

    assert result[ID_COLUMN].tolist() == ["a", "b"]


def test_drop_neutral_reviews_removes_score_three() -> None:
    """A nota 3 é ambígua e fica fora do problema binário."""
    reviews = make_reviews([("a", 3, "Ok"), ("b", 1, "Ruim"), ("c", 5, "Ótimo")])

    result = drop_neutral_reviews(reviews)

    assert result[SCORE_COLUMN].tolist() == [1, 5]


@pytest.mark.parametrize(
    ("score", "expected"),
    [(1, NEGATIVE), (2, NEGATIVE), (4, POSITIVE), (5, POSITIVE)],
)
def test_add_sentiment_label_maps_score_to_sentiment(score: int, expected: int) -> None:
    """Notas 1–2 são negativas e 4–5 são positivas."""
    reviews = make_reviews([("a", score, "texto")])

    result = add_sentiment_label(reviews)

    assert result[SENTIMENT_COLUMN].tolist() == [expected]


@pytest.mark.parametrize("invalid_score", [3, 0, 6])
def test_add_sentiment_label_rejects_scores_without_label(invalid_score: int) -> None:
    """Falhar alto é melhor que gerar um rótulo errado em silêncio."""
    reviews = make_reviews([("a", invalid_score, "texto")])

    with pytest.raises(ValueError, match=str(invalid_score)):
        add_sentiment_label(reviews)


def test_prepare_sentiment_dataset_applies_all_steps() -> None:
    """O pipeline completo filtra, deduplica, rotula e devolve só as colunas úteis."""
    reviews = make_reviews(
        [
            ("a", 5, "Ótimo"),
            ("a", 5, "Ótimo"),
            ("b", 3, "Ok"),
            ("c", 1, None),
            ("d", 2, "Atrasou"),
        ]
    )

    result = prepare_sentiment_dataset(reviews)

    assert result.columns.tolist() == [ID_COLUMN, MESSAGE_COLUMN, SCORE_COLUMN, SENTIMENT_COLUMN]
    assert result[ID_COLUMN].tolist() == ["a", "d"]
    assert result[SENTIMENT_COLUMN].tolist() == [POSITIVE, NEGATIVE]
    assert result.index.tolist() == [0, 1]


def test_prepare_sentiment_dataset_does_not_modify_input() -> None:
    """Quem chama continua com os dados originais para outras análises."""
    reviews = make_reviews([("a", 5, "Ótimo"), ("b", 3, "Ok"), ("c", 1, None)])
    original = reviews.copy()

    prepare_sentiment_dataset(reviews)

    pd.testing.assert_frame_equal(reviews, original)


def make_labeled_dataset(n_positive: int, n_negative: int) -> pd.DataFrame:
    """Monta um conjunto já rotulado, no formato devolvido por `prepare_sentiment_dataset`."""
    labels = [POSITIVE] * n_positive + [NEGATIVE] * n_negative
    return pd.DataFrame(
        {
            ID_COLUMN: [f"r{index}" for index in range(len(labels))],
            MESSAGE_COLUMN: [f"texto {index}" for index in range(len(labels))],
            SCORE_COLUMN: [5 if label == POSITIVE else 1 for label in labels],
            SENTIMENT_COLUMN: labels,
        }
    )


@pytest.fixture
def labeled_dataset() -> pd.DataFrame:
    """80 positivas e 20 negativas: proporção desbalanceada como a dos dados reais."""
    return make_labeled_dataset(n_positive=80, n_negative=20)


def test_split_train_test_uses_requested_test_size(labeled_dataset: pd.DataFrame) -> None:
    """20% do conjunto vai para teste e o restante para treino."""
    split = split_train_test(labeled_dataset, test_size=0.2)

    assert len(split.test) == 20
    assert len(split.train) == 80


def test_split_train_test_keeps_class_proportion(labeled_dataset: pd.DataFrame) -> None:
    """Com estratificação, a proporção é exata para qualquer semente, e não por sorte."""
    expected_positive_share = 0.8

    for seed in range(10):
        split = split_train_test(labeled_dataset, test_size=0.2, random_state=seed)

        assert split.train[SENTIMENT_COLUMN].mean() == pytest.approx(expected_positive_share)
        assert split.test[SENTIMENT_COLUMN].mean() == pytest.approx(expected_positive_share)


def test_split_train_test_has_no_overlap_and_covers_all_rows(
    labeled_dataset: pd.DataFrame,
) -> None:
    """Uma avaliação no treino e no teste ao mesmo tempo inflaria a métrica."""
    split = split_train_test(labeled_dataset)
    train_ids = set(split.train[ID_COLUMN])
    test_ids = set(split.test[ID_COLUMN])

    assert train_ids.isdisjoint(test_ids)
    assert train_ids | test_ids == set(labeled_dataset[ID_COLUMN])


def test_split_train_test_is_reproducible_with_same_seed(labeled_dataset: pd.DataFrame) -> None:
    """Mesma semente, mesma divisão: os resultados podem ser repetidos e comparados."""
    first = split_train_test(labeled_dataset, random_state=7)
    second = split_train_test(labeled_dataset, random_state=7)

    pd.testing.assert_frame_equal(first.train, second.train)
    pd.testing.assert_frame_equal(first.test, second.test)
