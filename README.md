# Поиск статей справки Авито

Решение ранжирует до 10 статей для каждого пользовательского запроса. Весь
pipeline находится в `solution.ipynb`.

## Идея решения

1. HTML статей очищается, длинные повторяющиеся блоки удаляются.
2. TF-IDF и BM25 находят точные совпадения, а `multilingual-e5-small` — близкие
   по смыслу фрагменты статей.
3. Лучшие 50 кандидатов повторно ранжируются с помощью
   `HistGradientBoostingClassifier`.
4. Query-kNN использует ответы похожих размеченных запросов.
5. Итоговый top-10 формируется объединением retrieval, reranker и query-kNN
   через RRF.

## Запуск

Требуется Python 3.13. Исходные файлы `articles.f`, `calibration.f` и `test.f`
должны находиться в папке `data/`.

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

В открывшемся notebook нужно выполнить **Run All**. Итоговый файл будет
сохранён в `output/answer.csv`.

## Оценка результата

На пяти semantic-grouped folds решение получает **0.613123 MAP@10**; результат
на leaderboard — около **0.59 MAP@10**.

Это практический потолок при текущей разметке, а не теоретический предел задачи.
Основное ограничение — небольшой объём и неоднозначность обучающих данных:
похожие запросы могут иметь разные наборы релевантных статей. Расширение
кандидатного пула и более сложные reranker, признаки, способы смешивания и
синтетические запросы не дали устойчивого прироста на честной групповой
валидации. Для заметного дальнейшего улучшения нужна прежде всего дополнительная
и более согласованная разметка.
