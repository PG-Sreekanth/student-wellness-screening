# Deployment Artifact Compatibility Note

This note documents how the project reduces model artifact loading issues between the ML training environment and the Streamlit or API runtime.

## Why this note exists

Serialized scikit-learn pipelines can fail to load if runtime versions differ from the versions used during training. Common failure patterns include:
- missing internal NumPy module paths during unpickling
- scikit-learn fitted-state validation changes across versions
- missing internal attributes on transformers such as `SimpleImputer`

## Current project approach

### 1) Version pinning for core serialization dependencies
The project pins the main packages that affect joblib artifact compatibility:

- `numpy==2.1.3`
- `scikit-learn==1.7.2`
- `joblib==1.4.2`
- `pandas==2.2.3`
- `catboost==1.2.8`

### 2) Runtime compatibility shim in `app/ml_predictor.py`
The app includes a small compatibility layer that:
- maps `numpy._core` imports to the available runtime modules when needed
- patches known fitted-state compatibility issues for some transformers
- loads the saved preprocessor and model artifacts lazily at startup

This is a practical deployment safeguard for Streamlit Cloud and local environments.

## Label alignment with SQL-cleaned logic
The application prediction layer is aligned with the SQL-cleaned labeling rules used in the project dataset:

### Wellness mapping
```sql
CASE
  WHEN red_flag <= 2 THEN 'high'
  WHEN red_flag <= 5 THEN 'moderate'
  ELSE 'low'
END
```

### Support priority mapping
- `critical`: `depression = 1` and `red_flag >= 6`
- `high priority`: `depression = 1` and `red_flag >= 5`
- `moderate priority`: `depression = 1` (remaining)
- `preventive high risk`: `depression = 0` and `red_flag >= 5`
- `preventive watchlist`: `depression = 0` and `red_flag >= 3`
- `stable`: all remaining cases

## Recommended practice for future retraining

When retraining the model:
1. Train and export artifacts in a clean, version-controlled environment.
2. Keep `requirements.txt` and `ml_pipeline/requirements.txt` updated together.
3. Save the final threshold configuration (`threshold_config.json`) with the model artifacts.
4. Run a quick app-side smoke test after replacing artifacts.
5. Re-check prediction label alignment against SQL and dashboard logic.

## Bottom line

The project now uses pinned versions plus a lightweight runtime compatibility shim to reduce artifact loading failures and keep deployment behavior consistent with the ML training outputs.
