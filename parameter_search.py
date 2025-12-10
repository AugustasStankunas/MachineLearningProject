from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV, KFold
from scipy.stats import randint, uniform, loguniform
from xgboost import XGBRegressor
import numpy as np
import pandas as pd

from sklearn.metrics import r2_score, make_scorer

def r2_eur_from_log(y_true_log, y_pred_log):
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    return r2_score(y_true, y_pred)

eur_r2_scorer = make_scorer(r2_eur_from_log, greater_is_better=True)


def run_param_search(X_train, X_val, y_train, y_val, *, random_state=42):
    xgb_base = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=5000,
        random_state=random_state,
        n_jobs=1,
        early_stopping_rounds=200
    )

    param_dist = {
        "max_depth": randint(3, 10),
        "min_child_weight": loguniform(1, 50),
        "gamma": uniform(0, 1.0),
        "subsample": uniform(0.5, 0.5),
        "colsample_bytree": uniform(0.5, 0.5),
        "reg_alpha": loguniform(1e-4, 1.0),
        "reg_lambda": loguniform(0.5, 10.0),
        "learning_rate": loguniform(0.01, 0.2),
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=random_state)

    search = HalvingRandomSearchCV(
        estimator=xgb_base,
        param_distributions=param_dist,
        factor=3,
        resource="n_estimators",
        max_resources=500,
        cv=cv,
        scoring=eur_r2_scorer,
        random_state=random_state,
        n_jobs=10,   # or 4 depending on RAM
        verbose=1,
    )

    X_full = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
    y_full = pd.concat([pd.Series(y_train), pd.Series(y_val)], axis=0).reset_index(drop=True)

    print("Starting hyperparameter search...")
    search.fit(
        X_full,
        y_full,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    best_params = {
        k: (v.item() if isinstance(v, np.generic) else v)
        for k, v in search.best_params_.items()
    }
    best_score = float(search.best_score_)

    print("Best params from search:", best_params)
    print("Best CV R² (EUR, via expm1 from log):", best_score)

    return best_params, best_score