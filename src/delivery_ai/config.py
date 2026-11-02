"""Constantes do projeto, centralizadas para notebook, testes e código usarem os mesmos valores."""

from pathlib import Path

# Caminhos calculados a partir deste arquivo, para funcionarem de qualquer pasta de execução.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
REVIEWS_PATH = DATA_DIR / "olist_order_reviews_dataset.csv"

ID_COLUMN = "review_id"
SCORE_COLUMN = "review_score"
MESSAGE_COLUMN = "review_comment_message"
SENTIMENT_COLUMN = "sentiment"

NEGATIVE = 0
POSITIVE = 1
SENTIMENT_NAMES = {NEGATIVE: "negativo", POSITIVE: "positivo"}

# Semente fixa para amostras e divisões aleatórias serem reproduzíveis.
RANDOM_STATE = 42

# 20% para teste: sobra volume para treinar e o teste ainda tem milhares de exemplos.
TEST_SIZE = 0.2

# Palavra precisa aparecer em pelo menos 2 avaliações do treino para virar coluna:
# as que aparecem uma vez só são quase sempre erros de digitação e não generalizam.
TFIDF_MIN_DF = 2

# A nota 3 fica fora do problema binário: é ambígua e viraria ruído nos rótulos.
NEUTRAL_SCORE = 3
NEGATIVE_SCORES = frozenset({1, 2})
POSITIVE_SCORES = frozenset({4, 5})
SCORE_TO_SENTIMENT = {
    **dict.fromkeys(NEGATIVE_SCORES, NEGATIVE),
    **dict.fromkeys(POSITIVE_SCORES, POSITIVE),
}
