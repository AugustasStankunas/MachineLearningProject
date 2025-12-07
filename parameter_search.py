from group import get_data
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV, KFold
from scipy.stats import randint, uniform, loguniform
from xgboost import XGBRegressor
import pandas as pd


X_train, X_test, X_val, y_train, y_test, y_val = get_data()

xgb_base = XGBRegressor(
    objective="reg:squarederror",
    n_estimators=5000,          # didelis, nes naudosim early stopping
    random_state=0,
    n_jobs=-1
)

param_dist = {
    "max_depth": randint(3, 10),
    "min_child_weight": loguniform(1, 50),
    "gamma": uniform(0, 1.0),
    "subsample": uniform(0.5, 0.5),        # 0.5–1.0
    "colsample_bytree": uniform(0.5, 0.5),
    "reg_alpha": loguniform(1e-4, 1.0),
    "reg_lambda": loguniform(0.5, 10.0),
    "learning_rate": loguniform(0.01, 0.2),
}

cv = KFold(n_splits=5, shuffle=True, random_state=42)

search = HalvingRandomSearchCV(
    estimator=xgb_base,
    param_distributions=param_dist,
    factor=3,
    resource="n_estimators",
    max_resources=2000,   # kiek max medžių leidi gero kandidato etapui
    cv=cv,
    scoring="r2",
    random_state=42,
    n_jobs=-1,
    verbose=1
)

X_full = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
y_full = pd.concat([pd.Series(y_train), pd.Series(y_val)], axis=0).reset_index(drop=True)

print("Starting hyperparameter search...")

search.fit(
    X_full, y_full,
    eval_set=[(X_val, y_val)],
    verbose=False,
    early_stopping_rounds=200
)

best_model = search.best_estimator_
print(search.best_params_, search.best_score_)