"""Compact search-space placeholders."""

EBM_SPACE = {"learning_rate": [0.02, 0.03, 0.05], "max_rounds": [80, 120, 180]}
XGBOOST_SPACE = {"max_depth": [2, 3], "learning_rate": [0.03, 0.05]}
RANDOM_FOREST_SPACE = {"max_depth": [5, 6, 8], "n_estimators": [160, 240]}
