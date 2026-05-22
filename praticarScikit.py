import json
from pathlib import Path
from lightgbm import LGBMRegressor
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from catboost import CatBoostRegressor


from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    AdaBoostRegressor,
)

from sklearn.linear_model import (
    Ridge,
    ElasticNet,
    Lasso,
    SGDRegressor,
)

from xgboost import XGBRegressor

from catboost import CatBoostRegressor

from lightgbm import LGBMRegressor


# =====================================================
# CONFIG
# =====================================================

ARQUIVO_PARQUET = "dados/cnpqBolsasAuxilios.parquet"
TARGET = "valor_pago"


# =====================================================
# MISSING VALUES
# =====================================================

NUMERIC_MISSING_VALUES = [-200, -200.0]

STRING_MISSING_VALUES = [
    "?", "NA", "N/A", "na", "n/a",
    "NaN", "nan", "", " ",
    "unknown", "Unknown", "-200",
]


# =====================================================
# HELPERS
# =====================================================

def make_one_hot_encoder():

    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=True
        )

    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=True
        )


def normalize_missing_values(df: pd.DataFrame):

    df = df.copy()

    for col in df.columns:

        if pd.api.types.is_numeric_dtype(df[col]):

            df[col] = df[col].replace(
                NUMERIC_MISSING_VALUES,
                np.nan
            )

        else:

            df[col] = (
                df[col]
                .astype("string")
                .str.strip()
                .replace(STRING_MISSING_VALUES, pd.NA)
            )

    return df


def build_preprocessor(X: pd.DataFrame, scale_numeric: bool):

    numeric_cols = X.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    categorical_cols = [
        c for c in X.columns
        if c not in numeric_cols
    ]

    return ColumnTransformer(
        transformers=[

            (
                "num",
                StandardScaler()
                if scale_numeric
                else "passthrough",
                numeric_cols
            ),

            (
                "cat",
                make_one_hot_encoder(),
                categorical_cols
            ),
        ]
    )


def evaluate_model(model, X_train, X_test, y_train, y_test):

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    mse = mean_squared_error(y_test, pred)

    return {
        "mae": float(mean_absolute_error(y_test, pred)),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_test, pred)),
    }


# =====================================================
# LOAD DATA
# =====================================================

print("\nCarregando dataset...")

df = pd.read_parquet(ARQUIVO_PARQUET)

print("Shape original:", df.shape)

# opcional -> amostra para acelerar
# df = df.sample(n=200000, random_state=42)

df = normalize_missing_values(df)

df = df.dropna()

print("Shape após dropna:", df.shape)

# =====================================================
# SPLIT
# =====================================================

X = df.drop(columns=[TARGET])

y = pd.to_numeric(
    df[TARGET],
    errors="coerce"
)

valid = y.notna()

X = X.loc[valid]
y = y.loc[valid]

# opcional -> log transform
# y = np.log1p(y)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTreino:", X_train.shape)
print("Teste :", X_test.shape)

# =====================================================
# RESULTS
# =====================================================

results = {}

# =====================================================
# RIDGE
# =====================================================

# print("\n" + "=" * 60)
# print("RIDGE")
# print("=" * 60)

# ridge_model = Pipeline([
#     (
#         "preprocess",
#         build_preprocessor(
#             X_train,
#             scale_numeric=True
#         )
#     ),
#     (
#         "model",
#         Ridge(alpha=1.0)
#     ),
# ])

# ridge_metrics = evaluate_model(
#     ridge_model,
#     X_train,
#     X_test,
#     y_train,
#     y_test
# )

# results["ridge"] = ridge_metrics

# print(ridge_metrics)

# =====================================================
# RANDOM FOREST TUNING
# =====================================================

# print("\n" + "=" * 60)
# print("RANDOM FOREST TUNING")
# print("=" * 60)

# rf_configs = [

#     {"n_estimators": 20, "max_depth": 10},

#     {"n_estimators": 50, "max_depth": 10},

#     {"n_estimators": 100, "max_depth": 15},

#     {"n_estimators": 200, "max_depth": 20},
# ]

# for config in rf_configs:

#     model_name = (
#         f"rf_"
#         f"trees{config['n_estimators']}_"
#         f"depth{config['max_depth']}"
#     )

#     print(f"\n[MODEL] {model_name}")

#     model = Pipeline([
#         (
#             "preprocess",
#             build_preprocessor(
#                 X_train,
#                 scale_numeric=False
#             )
#         ),

#         (
#             "model",
#             RandomForestRegressor(
#                 n_estimators=config["n_estimators"],
#                 max_depth=config["max_depth"],
#                 n_jobs=-10,
#                 random_state=42
#             )
#         ),
#     ])

#     metrics = evaluate_model(
#         model,
#         X_train,
#         X_test,
#         y_train,
#         y_test
#     )

#     results[model_name] = metrics

#     print(metrics)

# =====================================================
# CATBOOST TUNING
# =====================================================

# print("\n" + "=" * 60)
# print("CATBOOST TUNING")
# print("=" * 60)

# cat_configs = [

#     {
#         "iterations": 200,
#         "depth": 6,
#         "learning_rate": 0.05
#     },

#     {
#         "iterations": 300,
#         "depth": 8,
#         "learning_rate": 0.03
#     },

#     {
#         "iterations": 500,
#         "depth": 10,
#         "learning_rate": 0.02
#     },
# ]

# for config in cat_configs:

#     model_name = (
#         f"catboost_"
#         f"iter{config['iterations']}_"
#         f"depth{config['depth']}"
#     )

#     print(f"\n[MODEL] {model_name}")

#     model = Pipeline([
#         (
#             "preprocess",
#             build_preprocessor(
#                 X_train,
#                 scale_numeric=False
#             )
#         ),

#         (
#             "model",
#             CatBoostRegressor(
#                 iterations=config["iterations"],
#                 depth=config["depth"],
#                 learning_rate=config["learning_rate"],
#                 thread_count=-1,
#                 random_state=42,
#                 verbose=0,
            
#             )
#         ),
#     ])

#     metrics = evaluate_model(
#         model,
#         X_train,
#         X_test,
#         y_train,
#         y_test
#     )

#     results[model_name] = metrics

#     print(metrics)


# =====================================================
# XGBOOST TUNING
# =====================================================

# print("\n" + "=" * 60)
# print("XGBOOST TUNING")
# print("=" * 60)

# xgb_configs = [

#     {
#         "n_estimators": 200,
#         "max_depth": 6,
#         "learning_rate": 0.05
#     },

#     {
#         "n_estimators": 300,
#         "max_depth": 8,
#         "learning_rate": 0.03
#     },

#     {
#         "n_estimators": 500,
#         "max_depth": 10,
#         "learning_rate": 0.02
#     },
# ]

# for config in xgb_configs:

#     model_name = (
#         f"xgboost_"
#         f"iter{config['n_estimators']}_"
#         f"depth{config['max_depth']}"
#     )

#     print(f"\n[MODEL] {model_name}")

#     model = Pipeline([
#         (
#             "preprocess",
#             build_preprocessor(
#                 X_train,
#                 scale_numeric=False
#             )
#         ),

#         (
#             "model",
#             XGBRegressor(
#                 n_estimators=config["n_estimators"],
#                 max_depth=config["max_depth"],
#                 learning_rate=config["learning_rate"],
#                 subsample=0.8,
#                 colsample_bytree=0.8,
#                 n_jobs=-1,
#                 random_state=42
#             )
#         ),
#     ])

#     metrics = evaluate_model(
#         model,
#         X_train,
#         X_test,
#         y_train,
#         y_test
#     )

#     results[model_name] = metrics

#     print(metrics)


# =====================================================
# LIGHTGBM TUNING
# =====================================================

# print("\n" + "=" * 60)
# print("LIGHTGBM TUNING")
# print("=" * 60)

# lgbm_configs = [

#     {
#         "n_estimators": 200,
#         "max_depth": 6,
#         "learning_rate": 0.05,
#         "num_leaves": 31
#     },

#     {
#         "n_estimators": 300,
#         "max_depth": 8,
#         "learning_rate": 0.03,
#         "num_leaves": 50
#     },

#     {
#         "n_estimators": 500,
#         "max_depth": 10,
#         "learning_rate": 0.02,
#         "num_leaves": 100
#     },
# ]

# for config in lgbm_configs:

#     model_name = (
#         f"lightgbm_"
#         f"iter{config['n_estimators']}_"
#         f"depth{config['max_depth']}"
#     )

#     print(f"\n[MODEL] {model_name}")

#     model = Pipeline([
#         (
#             "preprocess",
#             build_preprocessor(
#                 X_train,
#                 scale_numeric=False
#             )
#         ),

#         (
#             "model",
#             LGBMRegressor(
#                 n_estimators=config["n_estimators"],
#                 learning_rate=config["learning_rate"],
#                 max_depth=config["max_depth"],
#                 num_leaves=config["num_leaves"],
#                 n_jobs=-8,
#                 random_state=42,
#                 verbose=-1
#             )
#         ),
#     ])

#     metrics = evaluate_model(
#         model,
#         X_train,
#         X_test,
#         y_train,
#         y_test
#     )

#     results[model_name] = metrics

#     print(metrics)

# # =====================================================
# # RESULTS TABLE
# # =====================================================

# print("\n" + "=" * 60)
# print("RESULTADOS FINAIS")
# print("=" * 60)

# results_df = pd.DataFrame(results).T

# results_df = results_df.sort_values(
#     by="r2",
#     ascending=False
# )

# print(results_df)

# =====================================================
# LINEAR MODELS
# =====================================================

# print("\n" + "=" * 60)
# print("LINEAR MODELS")
# print("=" * 60)

# linear_models = {

#     "lasso": Lasso(alpha=0.1),

#     "elasticnet": ElasticNet(
#         alpha=0.1,
#         l1_ratio=0.5
#     ),

#     "sgd": SGDRegressor(
#         max_iter=200,
#         tol=1e-3,
#         random_state=42
#     ),
# }

# for model_name, regressor in linear_models.items():

#     print(f"\n[MODEL] {model_name}")

#     model = Pipeline([

#         (
#             "preprocess",
#             build_preprocessor(
#                 X_train,
#                 scale_numeric=True
#             )
#         ),

#         (
#             "model",
#             regressor
#         ),
#     ])

#     metrics = evaluate_model(
#         model,
#         X_train,
#         X_test,
#         y_train,
#         y_test
#     )

#     results[model_name] = metrics

#     print(metrics)


# =====================================================
# EXTRA TREES
# =====================================================

# print("\n" + "=" * 60)
# print("EXTRA TREES")
# print("=" * 60)

# extra_configs = [

#     {
#         "n_estimators": 100,
#         "max_depth": 10
#     },

#     {
#         "n_estimators": 200,
#         "max_depth": 15
#     },
# ]

# for config in extra_configs:

#     model_name = (
#         f"extratrees_"
#         f"iter{config['n_estimators']}_"
#         f"depth{config['max_depth']}"
#     )

#     print(f"\n[MODEL] {model_name}")

#     model = Pipeline([

#         (
#             "preprocess",
#             build_preprocessor(
#                 X_train,
#                 scale_numeric=False
#             )
#         ),

#         (
#             "model",
#             ExtraTreesRegressor(
#                 n_estimators=config["n_estimators"],
#                 max_depth=config["max_depth"],
#                 n_jobs=-1,
#                 random_state=42
#             )
#         ),
#     ])

#     metrics = evaluate_model(
#         model,
#         X_train,
#         X_test,
#         y_train,
#         y_test
#     )

#     results[model_name] = metrics

#     print(metrics)



# =====================================================
# ADABOOST
# =====================================================

print("\n" + "=" * 60)
print("ADABOOST")
print("=" * 60)

ada_configs = [

    {
        "n_estimators": 100,
        "learning_rate": 0.05
    },

    {
        "n_estimators": 200,
        "learning_rate": 0.03
    },
]

for config in ada_configs:

    model_name = (
        f"adaboost_"
        f"iter{config['n_estimators']}"
    )

    print(f"\n[MODEL] {model_name}")

    model = Pipeline([

        (
            "preprocess",
            build_preprocessor(
                X_train,
                scale_numeric=True
            )
        ),

        (
            "model",
            AdaBoostRegressor(
                n_estimators=config["n_estimators"],
                learning_rate=config["learning_rate"],
                random_state=42
            )
        ),
    ])

    metrics = evaluate_model(
        model,
        X_train,
        X_test,
        y_train,
        y_test
    )

    results[model_name] = metrics

    print(metrics)

# =====================================================
# SAVE RESULTS
# =====================================================

Path("data").mkdir(exist_ok=True)



with open(
    "data/resultados_modelos.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        indent=2,
        ensure_ascii=False
    )

print("\nResultados salvos em:")
print("- data/resultados_modelos.csv")
print("- data/resultados_modelos.json")