import polars as pl

def adapt_input(df):
     # --- Encode Data ---
    cat_cols = df.select(pl.col(pl.Utf8)).columns

    df = df.with_columns(
        pl.col(c).cast(pl.Categorical).to_physical().alias(f"{c}_encoded") for c in cat_cols
    ).drop(cat_cols)

    return df