# delivery-ai: assistente de delivery com IA

Projeto de portfólio construído em 5 fases. Cada fase evolui a anterior e precisa provar a melhoria com métricas.

| Fase | O que faz | Status |
|---|---|---|
| **v1** | Classifica avaliações de clientes como positivas/negativas (TF-IDF + regressão logística) | ✅ concluída |
| v2 | LLM (Gemini) extrai o *motivo* das avaliações negativas, com saída JSON validada | próxima |
| v3 | Busca semântica de pratos por descrição ("algo leve com frango até 40 reais") | — |
| v4 | Agente que monta o pedido com chamada de funções | — |
| v5 | API, interface, Docker, CI e deploy | — |

---

## v1: sentimento das avaliações com ML clássico

**Pergunta:** dá para identificar, só pelo texto, quando um cliente está insatisfeito?

**Dados:** avaliações reais do [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (arquivo `olist_order_reviews_dataset.csv`).
- São 99.224 avaliações, mas só 41% têm texto.
- Depois da limpeza (texto obrigatório, sem duplicatas, sem a nota 3, que é ambígua) sobram **37.113**: notas 1–2 = negativo, 4–5 = positivo, sendo 71% positivas.

### Resultado

No conjunto de teste (7.423 avaliações que o modelo nunca viu):

| | Baseline (sempre "positivo") | **Modelo v1** |
|---|---|---|
| Acurácia | 71,0% | **93,1%** |
| Precisão (negativo) | 0,000 | **0,883** |
| Recall (negativo) | 0,000 | **0,878** |
| **F1 (negativo)** | 0,000 | **0,881** |
| F1 macro | 0,415 | **0,916** |

- O modelo encontra **88% dos clientes insatisfeitos**. Quando aponta um, acerta 88% das vezes.
- O baseline mostra por que a acurácia sozinha engana: chutar sempre "positivo" já dá 71% e não encontra nenhum insatisfeito.
- Treina em ~0,6 s e classifica 7,4 mil textos em ~0,4 s, numa CPU comum, sem GPU nem API.

### Como chegamos lá

1. **Exploração:** só 41% das avaliações têm texto, e as notas são desbalanceadas.
2. **Limpeza e rótulo:** a nota vira o rótulo; a nota 3 é descartada.
3. **Treino/teste 80/20 com `stratify`:** mesma proporção de classes nas duas partes.
4. **Baseline** com `DummyClassifier`: o piso a superar.
5. **TF-IDF + regressão logística** num único `Pipeline`. O vocabulário é aprendido só no treino, sem vazamento.
6. **Experimentos com validação cruzada** (só no treino; o teste é usado uma única vez, no fim).
7. **Análise de erros:** 20 erros lidos e classificados à mão.

### Experimentos: uma mudança por vez

Validação cruzada de 5 partes no treino. Critério definido antes de ver os resultados: F1 negativo.

| Configuração | Recall neg. | Precisão neg. | F1 neg. |
|---|---|---|---|
| Palavras soltas (inicial) | 0,870 | 0,878 | 0,874 |
| **+ bigramas** ✅ escolhida | 0,880 | 0,883 | **0,881** |
| + `class_weight="balanced"` | 0,941 | 0,821 | 0,877 |
| bigramas + balanced | 0,946 | 0,825 | 0,881 |

- **Bigramas** ("nao recebi", "nao tive") melhoraram tudo um pouco, sem custo.
- **`balanced`** não é uma melhoria, é uma troca: acha 7 pontos a mais de insatisfeitos em troca de 6 pontos a mais de alarmes falsos. Fica como decisão de negócio.

### O que os erros ensinam

Amostra de 20 erros, lida e classificada à mão:

| Tipo de erro | % |
|---|---|
| A nota não bate com o texto (ex.: nota 1 com "Excelente, muito satisfeito") | 35% |
| Elogio e reclamação no mesmo texto | 20% |
| Reclamação sem palavras negativas fortes ("a empresa deveria testar o produto antes") | 20% |
| Sem opinião clara (perguntas, informações) | 15% |
| Negação ("Ótima entrega não demorou nada!" → negativo) | 5% |
| Texto sem conteúdo (".") | 5% |

**Limites do modelo clássico:**
- ele soma palavras soltas, sem entender o contexto;
- diz *se* o cliente está insatisfeito, mas não *por quê*;
- aprende com notas que às vezes contradizem o texto.

Um exemplo: "chegou atrasado e amassado" sai como **positivo** (44% de chance de negativo). "Amassado" e "atrasado" pesam pouco, porque muitos clientes dão nota alta mesmo citando isso, e a tendência para "positivo" (71% dos dados) vence. Esses limites motivam a v2.

### Como rodar

```bash
# 1. Ambiente (Python 3.13)
python -m venv .venv
.venv\Scripts\activate          # Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# 2. Dados: baixe o dataset no Kaggle (link acima) e coloque
#    olist_order_reviews_dataset.csv em data/

# 3. Treinar (salva models/sentiment_model.joblib e mostra as métricas)
python -m delivery_ai.train

# 4. Classificar frases
python -m delivery_ai.predict "adorei, chegou antes do prazo" "produto veio quebrado"

# 5. Testes e lint
pytest
ruff check . && ruff format --check .
```

A análise completa, com gráficos, está no notebook [`notebooks/v1_sentimento.ipynb`](notebooks/v1_sentimento.ipynb).

### Estrutura

```
src/delivery_ai/
  config.py       constantes centralizadas (caminhos, colunas, configuração escolhida)
  data.py         carregamento, limpeza, rotulagem e divisão treino/teste
  models.py       construtores do baseline, do TF-IDF e do pipeline
  evaluation.py   métricas focadas na classe negativa e validação cruzada
  classifier.py   SentimentClassifier: treina, prevê, salva e carrega
  train.py        script de treino (python -m delivery_ai.train)
  predict.py      script de previsão (python -m delivery_ai.predict)
tests/            testes com pytest, escritos antes do código (TDD)
notebooks/        exploração e análise
```

### Decisões técnicas

- **`Pipeline` TF-IDF → regressão logística:** um único objeto que aprende o vocabulário só no treino e é salvo inteiro num arquivo.
- **Regressão logística:** rápida, forte em texto e interpretável (dá para ver o peso de cada palavra).
- **Sem lista de stopwords:** as listas comuns em português removem o "não", que inverte o sentido da frase.
- **Validação cruzada para escolher, teste para confirmar:** escolher olhando o teste deixaria a métrica final otimista.
- **`SentimentClassifier` recebe o modelo pelo construtor:** facilita testes e a troca de modelo no futuro.

---

## Próximo: v2 com LLM

Usar o Gemini para **extrair o motivo** das avaliações negativas (atraso, produto errado, item faltando...) com saída JSON validada por Pydantic, e comparar com a v1 em **acerto, custo e latência**.
