def filter_predictions(original_df):
    df_pd = original_df.to_pandas()

    # Selecting only summer sales
    df_summer = df_pd[(df_pd['mes'] == 1) | (df_pd['mes'] == 2) | (df_pd['mes'] == 12)]

    # Sum of values
    sum_of_sales = df_summer.groupby(by=['pdv', 'sku'])['vendas_semanais'].sum()
    sum_of_sales = sum_of_sales.sort_values(ascending=False)
    
    sum_of_sales = sum_of_sales.round()

    return sum_of_sales[:300000].index


from datetime import datetime
import polars as pl

def filter_most_valuable_inputs(df, n):
    """
        Use this on the dataframe served as history for the predictions.
        This function will return only the top n SKU x PDV combinations
        with highest gross profit value and their total values during january, february and march.
    """

    summer_df = df.filter(pl.col('transaction_date').is_between(datetime(2022, 1, 1), datetime(2022, 2, 28)) | pl.col('transaction_date').is_between(datetime(2022, 12, 1), datetime(2022, 12, 31)))
    total_profit_per_comb = summer_df[:, ['internal_store_id', 'internal_product_id', 'gross_value']].group_by(['internal_store_id', 'internal_product_id']).sum()
    total_profit_per_comb = total_profit_per_comb.sort(['gross_value'], descending=True)
    return total_profit_per_comb.head(n)
