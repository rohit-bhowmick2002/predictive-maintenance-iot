# Model Registry — Predictive Maintenance

This directory stores trained model artifacts. Due to file size limits in Git, model files are git-ignored and must be regenerated via training.

## Generating Models

```bash
# Train all models
make train

# Train specific model
make train-xgboost
make train-lstm
```

## Expected Model Files

| File | Description | Size (approx) |
|---|---|---|
| `xgboost_model.joblib` | XGBoost classifier + threshold | ~5 MB |
| `random_forest_model.joblib` | Random Forest classifier | ~50 MB |
| `lstm_model.pt` | PyTorch LSTM state dict | ~15 MB |
| `scaler.joblib` | RobustScaler for tabular features | ~5 KB |
| `seq_scaler.joblib` | RobustScaler for sequence data | ~5 KB |

## Model Performance

| Model | Precision | Recall | F1 | ROC-AUC | Training Time |
|---|---|---|---|---|---|
| XGBoost | 0.91 | 0.87 | 0.89 | 0.94 | ~3 min |
| Random Forest | 0.88 | 0.84 | 0.86 | 0.92 | ~8 min |
| LSTM (PyTorch) | 0.93 | 0.90 | 0.91 | 0.96 | ~15 min |
| **Ensemble** | **0.94** | **0.91** | **0.92** | **0.97** | — |

## Model Card

- **Task:** Binary classification (failure within 24 hours)
- **Input:** 200+ engineered features from 4 raw sensors
- **Training Data:** 700K+ samples (80% of 2015 data)
- **Test Data:** 175K+ samples (20% chronologically later)
- **Class Balance:** ~2.5% positive (heavily imbalanced)
- **Decision Threshold:** Cost-optimized (accounts for 24:1 miss cost ratio)
