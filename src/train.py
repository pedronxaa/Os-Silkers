import pandas as pd
import polars as pl
import numpy as np
import lightgbm as lgb
from wmape import eval_wmape
from final_changes import adapt_input

def train(df_final:pl.DataFrame):

    df_final = adapt_input(df_final)

    df_final = df_final.to_pandas()
    df_final = df_final.replace([np.nan], [0])
    df_final = pl.from_pandas(df_final)

    # --- Define target and features ---
    COLUNAS_DROP = ["semana", "vendas_semanais"]
    y = df_final["vendas_semanais"]
    X = df_final.drop(COLUNAS_DROP)

    feature_names = X.columns

    # --- Convert to pandas (LGBM likes pandas) ---
    X = X.to_pandas()
    y = y.to_pandas()
    print(X, y)

    # --- Data splitting for forecasting ---
    # Get the last week in the dataframe
    ultima_semana = df_final.select(pl.col("semana").max()).item()  # tipo datetime.date
    # Define split: train up to 5 weeks before the last week
    semana_corte = pd.to_datetime(ultima_semana) - pd.Timedelta(weeks=5)

    # Create training and validation masks
    train_idx = df_final.get_column("semana").to_pandas() <= semana_corte
    val_idx = df_final.get_column("semana").to_pandas() > semana_corte

    # Apply masks
    X_train, X_val = X[train_idx].copy(), X[val_idx].copy()
    y_train, y_val = y[train_idx].copy(), y[val_idx].copy()

    # ENCODING ==================================================

    cat_cols_encoded = [c for c in X_train.columns if '_encoded' in c]

    for col in X_train.columns:
        if col in cat_cols_encoded:
            X_train[col] = X_train[col].astype('category')
            X_val[col] = X_val[col].astype('category')
        elif X_train[col].dtype == 'int64':
            X_train[col] = X_train[col].astype('int32')
            X_val[col] = X_val[col].astype('int32')

    # TRAINING ==================================================

    lgbm = lgb.LGBMRegressor(
        objective="regression_l1",
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=150,
        random_state=42,
        n_jobs=-1,
        colsample_bytree=0.8,
        subsample=0.8,
        max_bin=10000
    )

    lgbm.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric=eval_wmape,
        callbacks=[lgb.early_stopping(100, verbose=True)]
    )

    return lgbm
   