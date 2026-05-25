import pandas as pd
import numpy as np

FEATURES = ['km', 'condition', 'vehicle_age', 'num_revisions', 'engine_temperature']
TARGET = 'breakdown'

_FALLBACK_DATA = {
    'id': range(101, 111),
    'km':               [15000, 45000, 12000, 80000, 32000, 65000, 22000, 95000, 40000, 28000],
    'condition':        [2,     1,     2,     0,     1,     0,     2,     0,     1,     2],
    'vehicle_age':      [2,     8,     3,     15,    6,     18,    4,     22,    7,     5],
    'num_revisions':    [2,     7,     3,     12,    5,     14,    4,     18,    6,     4],
    'engine_temperature':[75,   88,    72,    108,   85,    112,   73,    115,   90,    76],
    'breakdown':        [0,     0,     0,     1,     0,     1,     0,     1,     0,     0],
}


def load_data(path='data/raw_data_vab.csv'):
    try:
        df = pd.read_csv(path)
        print(f"Dataset loaded: {df.shape[0]} rows.")
        return df
    except FileNotFoundError:
        print("CSV not found. Generating fallback dataset...")
        return pd.DataFrame(_FALLBACK_DATA)


def _remove_duplicates(df):
    return df.drop_duplicates(subset=['id'], keep='first')


def _convert_to_numeric(df):
    for col in FEATURES:
        df.loc[:, col] = pd.to_numeric(df[col], errors='coerce')
    return df


def _handle_missing_values(df):
    df = df.dropna(subset=[TARGET])
    if df['km'].isnull().all():
        print("WARNING: km column entirely empty, filling with 0")
        df['km'] = df['km'].fillna(0)
    else:
        df.loc[:, 'km'] = df['km'].fillna(df['km'].median())
    df['condition'] = df['condition'].fillna(2)
    df['vehicle_age'] = df['vehicle_age'].fillna(df['vehicle_age'].median())
    df['num_revisions'] = df['num_revisions'].fillna(df['num_revisions'].median())
    df['engine_temperature'] = df['engine_temperature'].fillna(df['engine_temperature'].median())
    return df


def _filter_outliers(df):
    return df[(df['km'] >= 0) & (df['km'] <= 1_000_000)]


def clean_data(df):
    df = _remove_duplicates(df)
    df = _convert_to_numeric(df)
    df = _handle_missing_values(df)
    df = _filter_outliers(df)
    return df
