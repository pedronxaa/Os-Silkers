import pandas as pd
import polars as pl
import lightgbm as lgb
import numpy as np
import matplotlib.pyplot as plt
from pyarrow.parquet import ParquetFile
import sklearn.model_selection as train_test_split
import sklearn.metrics as metrics
import seaborn as sns
import os


def load_history(dataset):

    # Loading ----------------
    #file_path = os.path.join(parquet_files_path, 'associados_enriquecido.parquet')
    #dataset = pl.read_parquet(file_path)

    # Transforming negative values into 0's
    dataset = dataset.with_columns([
        pl.when(pl.col("quantity") < 0)
        .then(0)
        .otherwise(pl.col("quantity"))
        .alias("quantity")
    ])


    # Group by PDV x SKU x Transaction Date ----------------------------------
    df_semanal = dataset.group_by([
        pl.col("internal_store_id").alias("pdv"),
        pl.col("internal_product_id").alias("sku"),
        pl.col("transaction_date").dt.truncate("1w").alias("semana")
        ]).agg(
            pl.sum("quantity").alias("vendas_semanais")).sort("semana")
    
    # Adding Lag and Rolling Average / Standard Deviation --------------------
    df_features = df_semanal.sort(["pdv", "sku", "semana"]).with_columns([

        # Adding Lag
        pl.col("vendas_semanais").shift(1).over(["pdv", "sku"]).alias("vendas_lag_1"),
        pl.col("vendas_semanais").shift(2).over(["pdv", "sku"]).alias('vendas_lag_2'),
        pl.col("vendas_semanais").shift(3).over(["pdv", "sku"]).alias('vendas_lag_3'),
        pl.col("vendas_semanais").shift(4).over(["pdv", "sku"]).alias('vendas_lag_4'),
        
        # Adding rolling mean and std
        pl.col("vendas_semanais").shift(1).rolling_mean(window_size=4).over(["pdv", "sku"]).alias("media_movel_4_semanas"),
        pl.col("vendas_semanais").shift(1).rolling_std(window_size=4).over(["pdv", "sku"]).alias("desvio_padrao_4_semanas"),
        
        # Adding date columns
        pl.col("semana").dt.week().alias("semana_do_ano"),
        pl.col("semana").dt.month().alias("mes"),
        pl.col("semana").dt.year().alias("ano"),
    ])

    #df_features = df_features.with_columns(
    #    pl.when(pl.col("vendas_semanais") > 0).then(pl.col("semana_do_ano")).otherwise(-1).forward_fill().over(["pdv", "sku"]).alias("ultima_semana_com_venda")
    #)

    df_final = df_features
    #df_final = df_features.drop_nulls()

    return df_final, df_features