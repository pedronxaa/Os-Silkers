from load_history import load_history
from train import train
from datetime import date, datetime, timedelta
import polars as pl
import pandas as pd
from final_changes import adapt_input
import numpy as np

# input features: vendas_lag_1, ..., vendas_lag_4, media_movel_4_semanas, desvio_padrao_4_semanas,
# semana_do_ano, mes, ano, pdv_encoded, sku_encoded

def predict(day, month, year, model, history_dataset, save_path='.', num_weeks=5):
    """
        Returns a dataframe of predictions for the next num_weeks, starting from day-month-year.
    """

    original_df = history_dataset
    # Step 1: Get Lag Dates ==========================================================

    # Extra data
    #date_to_predict= date(year=2023, month=1, day=2)
    initial_date_to_predict = datetime(year=year, month=month, day=day)

    #df = df.to_pandas()
    original_df = original_df.to_pandas()

    # Step 1: Get lag dates ==========================================================

    concat_df = None
    to_predict_columns = ['semana', 'pdv', 'produto', 'quantidade']
    prediction_results_list = []

    for week_i in range(num_weeks):
        date_to_predict = initial_date_to_predict + timedelta(week_i * 7)
        date1 = date_to_predict - timedelta(7)
        date2 = date1 - timedelta(7)
        date3 = date2 - timedelta(7)
        date4 = date3 - timedelta(7)

        # Step 2: Get prediction dates for a single week =================================
        week_to_predict = date_to_predict.isocalendar().week


        new_original_df = pd.DataFrame()

        # Copy pdv x sku combinations
        new_original_df['pdv'] = original_df['pdv'].copy(deep=False)
        new_original_df['sku'] = original_df['sku'].copy(deep=False)
        # Set unknown data as None
        new_original_df['semana'] = date_to_predict
        new_original_df['vendas_semanais'] = None
        new_original_df['vendas_lag_1'] = None
        new_original_df['vendas_lag_2'] = None
        new_original_df['vendas_lag_3'] = None
        new_original_df['vendas_lag_4'] = None
        new_original_df['media_movel_4_semanas'] = None
        new_original_df['desvio_padrao_4_semanas'] = None
        # Fill in with date to predict
        new_original_df['semana_do_ano'] = week_to_predict
        new_original_df['mes'] = date_to_predict.month
        new_original_df['ano'] = date_to_predict.year

        # Remove duplicate data
        new_original_df = new_original_df.drop_duplicates(subset=['pdv', 'sku'])

        # Step 3 =========================================================================

        # Once the concat is done, we must group everything to make sure pdv x sku combinations are in order (sorted by  pdv x sku, then by date)!
        # This is important for our lag calculation, as it will include our to-be-predicted samples where they need to be in order to get the last 5 week values for each combination.

        # We will only get this concatenation if 
        if week_i == 0:
            four_weeks_df = original_df[original_df['semana'] >= datetime(date4.year, date4.month, 1)]
            concat_df = pd.concat([four_weeks_df, new_original_df])
        else:
            concat_df = pd.concat([concat_df, new_original_df])

        # Step 4 =========================================================================
        concat_df = concat_df.sort_values(by=['pdv', 'sku', 'semana'])

        # Step 5: Compute Lag ============================================================

        # Adding Lag and Rolling Average / Standard Deviation --------------------
        concat_df_pl = pl.from_pandas(concat_df)

        concat_df_pl = concat_df_pl.with_columns([ # Sort ordena o dataframe pela coluna pdv, sku e semana

            # Adding Lag
            pl.col("vendas_semanais").shift(1).over(["pdv", "sku"]).alias("vendas_lag_1"), # Shift desloca a coluna vendas_semanais uma semana para frente
            pl.col("vendas_semanais").shift(2).over(["pdv", "sku"]).alias('vendas_lag_2'), # Shift desloca a coluna vendas_semanais duas semanas para frente
            pl.col("vendas_semanais").shift(3).over(["pdv", "sku"]).alias('vendas_lag_3'), # Shift desloca a coluna vendas_semanais três semanas para frente
            pl.col("vendas_semanais").shift(4).over(["pdv", "sku"]).alias('vendas_lag_4'), # Shift desloca a coluna vendas_semanais quatro semanas para frente
            
            # Adding rolling mean and std
            pl.col("vendas_semanais").shift(1).rolling_mean(window_size=4).over(["pdv", "sku"]).alias("media_movel_4_semanas"),
            pl.col("vendas_semanais").shift(1).rolling_std(window_size=4).over(["pdv", "sku"]).alias("desvio_padrao_4_semanas")
        ])

        concat_df = concat_df_pl.to_pandas()
        
        # Step 6: Generate Dataframe to Predict =============================================
        
        # Filter out dates for only the desired one
        to_predict_df = concat_df[concat_df['semana'] == date_to_predict]

        # Replace null values

        to_predict_df = to_predict_df.replace([pd.NA, np.nan], [0, 0])

        # Sort values
        to_predict_df = to_predict_df.sort_values(by=['semana'])

        # Change Column Order (if necessary)
        #to_predict_df[:,['pdv', 'sku', 'semana', 'vendas_lag1_', 'vendas_lag_2', 'vendas_lag_3', 'vendas_lag_4', 'media_movel_4_semanas', 'desvio_padrao_4_semanas', 'semana_do_ano', 'mes', 'ano']]

        # Step 7: Make Final Changes =========================================================

        to_predict_df = pl.from_pandas(to_predict_df)

        to_predict_df = adapt_input(to_predict_df)

        to_predict_df = to_predict_df.sort(by=['semana'])

        # Drop the column we want to predict ("vendas_semanais") and "semana", which is not part of the input
        to_predict_df = to_predict_df.drop(['semana', 'vendas_semanais'])

        # Step 8:  Run Model Prediction ======================================================

        print(f'Initializing predictions for {date_to_predict}')

        predictions = model.predict(to_predict_df)

        # Drop unnecessary values for next predictions
        concat_df = concat_df[concat_df['semana'] > date4]

        # Add predictions to table
        concat_df.loc[concat_df['semana'] == date_to_predict, 'vendas_semanais'] = np.round(predictions)

        # Add predictions to results dataframe
        predicted_values_df = concat_df.rename({
            'vendas_semanais': 'quantidade',
            'sku': 'produto'
        }, axis='columns').sort_values(by=['semana'])[concat_df['semana'] == date_to_predict][to_predict_columns]

        predicted_values_df['semana'] = week_i + 1

        prediction_results_list.append(predicted_values_df)
        print(f'End of predictions for {date_to_predict}')
        # No final da execução, temos:
        # df = date4, date3, date2, date1, date_to_predict, date_to_predict_2

    return pd.concat(prediction_results_list)