#arcenesis-magnit-uplift
hurdle t-learner на catboost.

## запуск
```bash
uv sync
uv run python -m src.train --train data/train.parquet --out artifacts/
uv run python -m src.predict --test data/test.parquet --models artifacts/ --out predictions.csv
```

## внутри
4 catboost модели:
- p_treat, p_ctrl: купит/не купит
- e_treat, e_ctrl: сумма при условии покупки
uplift = p_treat * e_treat - p_ctrl * e_ctrl

## валидация
5-fold stratified kfold по treatment_flg * 2 + is_buyer, метрика uplift@10 с бутстрэпом(нижняя граница 95% ди).
