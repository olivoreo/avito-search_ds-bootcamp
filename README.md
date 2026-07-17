# Поиск статей справки Авито

Локальное решение без внешних API: lexical retrieval дополняется dense retrieval по фрагментам статей, а итоговые top-50 кандидаты сортируются лёгким reranker из scikit-learn.

## Pipeline

1. Очистка HTML и удаление точных повторов длинных блоков.
2. Word/char TF-IDF и BM25 отдельно для `title` и `body`.
3. RRF lexical-ранжирование.
4. Разбиение статей на фрагменты до 192 токенов с overlap 24.
5. Локальные mean-pooled embeddings модели [`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small) (117.7M параметров, 384 измерения, MIT).
6. Для статьи объединяются scores двух лучших chunks: `0.8 × best + 0.2 × second`.
7. RRF `2×lexical + dense` формирует top-50 кандидатов.
8. `HistGradientBoostingClassifier` ранжирует кандидатов по scores и ranks TF-IDF, BM25, lexical RRF и E5.
9. Победивший pipeline генерирует и проверяет `output/answer.csv`.

E5 загружается только из локального Hugging Face cache. Тяжёлый neural cross-encoder удалён из финального решения. Лёгкий reranker обучается только на `calibration.f`: для локальной оценки используются честные out-of-fold предсказания, где validation-запросы не входят в обучение.

## Локальная оценка

| Метод | 5-fold MAP@10 | std | Recall@10 |
|---|---:|---:|---:|
| Lexical | 0.316691 | 0.037799 | 0.652000 |
| Dense E5-small | 0.283106 | 0.023532 | 0.590667 |
| RRF lexical + dense top-2 | 0.415933 | 0.032621 | 0.758333 |
| LogisticRegression reranker | 0.436705 | 0.021527 | 0.792833 |
| HistGradientBoosting reranker | **0.457765** | **0.013333** | **0.806833** |

Recall@50 candidate pool: **0.961000**.

Прирост MAP@10 относительно lexical baseline: **+0.141074**, или примерно **+44.5%**. GradientBoosting оказался лучше retrieval на каждом из пяти folds. Полный CPU-прогон после загрузки весов занимает несколько минут.

## Запуск

Требуется Python 3.13. После установки зависимостей один раз загрузите зафиксированные ревизии моделей:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

hf download intfloat/multilingual-e5-small \
  model.safetensors config.json tokenizer.json tokenizer_config.json \
  special_tokens_map.json sentencepiece.bpe.model \
  --revision 614241f622f53c4eeff9890bdc4f31cfecc418b3

jupyter lab solution.ipynb
```

Запустите Notebook командой **Run All**. Результат сохраняется в `output/answer.csv`.

## Файлы

- `solution.ipynb` — всё решение и выполненные результаты;
- `data/` — неизменённые исходные данные;
- `output/answer.csv` — итоговые ответы;
- `requirements.txt` — зависимости.
