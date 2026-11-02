"""Classificador de sentimento pronto para uso: treina, prevê, salva e carrega.

É a peça que sai do notebook para o código de produção. Scripts (e, na v5, a API)
usam esta classe sem precisar saber como o modelo é montado por dentro.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import joblib
from sklearn.pipeline import Pipeline

from delivery_ai.config import (
    NEGATIVE,
    SENTIMENT_CLASS_WEIGHT,
    SENTIMENT_MODEL_PATH,
    SENTIMENT_NAMES,
    SENTIMENT_NGRAM_RANGE,
)
from delivery_ai.models import build_sentiment_model


@dataclass(frozen=True)
class SentimentPrediction:
    """Resultado de uma previsão: o texto, o sentimento e a chance de ser negativo.

    Guarda o sentimento como número (para métricas) e expõe o nome (para pessoas).
    """

    text: str
    sentiment: int
    negative_probability: float

    @property
    def label(self) -> str:
        """Nome legível do sentimento ("negativo" ou "positivo")."""
        return SENTIMENT_NAMES[self.sentiment]


class SentimentClassifier:
    """Envolve o pipeline treinado e esconde os detalhes do scikit-learn de quem usa.

    Recebe o pipeline pronto pelo construtor, em vez de criá-lo, para que testes e versões
    futuras possam injetar outro modelo sem mudar quem usa a classe.
    """

    def __init__(self, model: Pipeline) -> None:
        self._model = model

    @classmethod
    def train(
        cls,
        texts: Sequence[str],
        labels: Sequence[int],
        ngram_range: tuple[int, int] = SENTIMENT_NGRAM_RANGE,
        class_weight: str | None = SENTIMENT_CLASS_WEIGHT,
    ) -> Self:
        """Treina a configuração escolhida na v1 e devolve o classificador pronto."""
        model = build_sentiment_model(ngram_range=ngram_range, class_weight=class_weight)
        return cls(model.fit(texts, labels))

    def predict(self, texts: Sequence[str]) -> list[SentimentPrediction]:
        """Classifica vários textos de uma vez, o que é bem mais rápido que um por um."""
        if len(texts) == 0:
            return []
        sentiments = self._model.predict(texts)
        negative_column = list(self._model.classes_).index(NEGATIVE)
        negative_probabilities = self._model.predict_proba(texts)[:, negative_column]
        return [
            SentimentPrediction(
                text=text, sentiment=int(sentiment), negative_probability=float(probability)
            )
            for text, sentiment, probability in zip(
                texts, sentiments, negative_probabilities, strict=True
            )
        ]

    def save(self, path: Path = SENTIMENT_MODEL_PATH) -> Path:
        """Grava o pipeline inteiro num arquivo só, para vetorizador e modelo nunca se separarem."""
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._model, path)
        return path

    @classmethod
    def load(cls, path: Path = SENTIMENT_MODEL_PATH) -> Self:
        """Carrega um modelo salvo.

        Só carregue arquivos de fonte confiável: o joblib usa pickle, que pode executar código.
        """
        if not path.exists():
            raise FileNotFoundError(
                f"Modelo não encontrado em {path}. "
                "Rode `python -m delivery_ai.train` para treinar e salvar o modelo."
            )
        return cls(joblib.load(path))
