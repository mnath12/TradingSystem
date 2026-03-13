"""
Example script demonstrating data cleaning and preparation.
This script shows how to clean existing CSV data files.
"""

import pandas as pd
from data import DataCleaner

def clean_existing_data():
    """
    Example: Clean data from an existing CSV file.
    """
    # Load raw data from CSV
    print("Loading data from CSV...")
    df_raw = pd.read_csv('aapl_intraday_data.csv')
    
    print(f"Raw data shape: {df_raw.shape}")
    print(f"Columns: {list(df_raw.columns)}")
    
    # Initialize cleaner
    cleaner = DataCleaner()
    
    # Clean and prepare data
    print("\n" + "=" * 60)
    print("Cleaning and preparing data...")
    print("=" * 60)
    
    df_clean = cleaner.prepare_data(
        df_raw,
        datetime_column='Datetime',
        add_returns=True,
        add_moving_averages=True,
        add_volatility=True,
        add_rsi=True,
        add_macd=True,
        add_bollinger_bands=True,
        add_price_features=True,
        add_volume_features=True
    )
    
    # Save cleaned data
    output_file = 'aapl_intraday_data_cleaned.csv'
    df_clean.to_csv(output_file)
    print(f"\nCleaned data saved to {output_file}")
    
    # Display summary
    print("\n" + "=" * 60)
    print("CLEANED DATA SUMMARY")
    print("=" * 60)
    print(f"\nShape: {df_clean.shape}")
    print(f"\nFirst few rows:")
    print(df_clean.head())
    
    return df_clean

if __name__ == "__main__":
    try:
        df_clean = clean_existing_data()
    except FileNotFoundError:
        print("Error: 'aapl_intraday_data.csv' not found.")
        print("Please run 'python data.py' first to download data, or provide your own CSV file.")
    except Exception as e:
        print(f"Error: {e}")

