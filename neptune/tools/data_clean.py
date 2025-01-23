import pandas as pd
import numpy as np
from scipy.stats import skew
import time
import warnings
from joblib import Parallel, delayed

def _drop_na_columns(dataset):
    threshold = 0.5 * len(dataset)
    return dataset.dropna(thresh=threshold, axis=1)

def _drop_na_rows(dataset):
    return dataset.dropna(axis=0)

def _check_skew(column):
    if column.isnull().any():
        column = column.dropna()
    return 0 if skew(column) == 0 else 1

def _impute_numeric(dataset):
    numeric_columns = dataset.select_dtypes(include=['number']).columns

    if numeric_columns.empty:
        return dataset

    mode_values = dataset[numeric_columns].mode()
    if mode_values.empty:
        return dataset

    mode_values = mode_values.iloc[0]
    mean_values = dataset[numeric_columns].mean()
    median_values = dataset[numeric_columns].median()

    def impute_column(col):
        if dataset[col].isnull().any():
            if (dataset[col] % 1 == 0).all():
                skewness = _check_skew(dataset[col])
                fill_value = mean_values[col] if skewness == 0 else median_values[col]
            else:
                fill_value = mean_values[col]
            dataset[col] = dataset[col].fillna(fill_value)
        return dataset

    for col in numeric_columns:
        dataset = impute_column(col)
    return dataset



def _impute_strings(dataset):
    string_columns = dataset.select_dtypes(include=['object']).columns

    if string_columns.empty:
        return dataset

    mode_values = dataset[string_columns].mode()
    if mode_values.empty:
        return dataset

    mode_values = mode_values.iloc[0]

    def impute_column(col):
        if dataset[col].isnull().any():
            unique_values = dataset[col].dropna().unique()
            if len(unique_values) == len(dataset[col].dropna()):
                dataset[col] = dataset[col].apply(
                    lambda x: np.random.choice(unique_values) if pd.isnull(x) else x
                )
            else:
                dataset[col] = dataset[col].fillna(mode_values[col])
        return dataset

    dataset = Parallel(n_jobs=-1)(delayed(impute_column)(col) for col in string_columns)
    return dataset[-1]


def _calculate_removed_rows_columns(original_shape, cleaned_shape):
    removed_rows = original_shape[0] - cleaned_shape[0]
    removed_columns = original_shape[1] - cleaned_shape[1]
    return {"removed_rows": removed_rows, "removed_columns": removed_columns}

def clean(dataset):
    warnings.filterwarnings("ignore")
    start_time = time.time()
    
    original_shape = dataset.shape
    null_value = dataset.isnull().sum().sum()
    null_info = dataset.isnull().sum()
    print(f"uncleaned dataset : \n{null_info}\n")
    print(f"{'shape':<15}{str(original_shape):>15}")
    print(f"{'null values':<15}{null_value:>15}\n")
    
    if null_value > 0:
        dataset = _drop_na_columns(dataset)
        dataset = _impute_numeric(dataset)
        dataset = _impute_strings(dataset)
        dataset = _drop_na_rows(dataset)
    
    cleaned_shape = dataset.shape
    removed_info = _calculate_removed_rows_columns(original_shape, cleaned_shape)
    null_value = dataset.isnull().sum().sum()
    null_info = dataset.isnull().sum()
    
    print(f"cleaned dataset : \n{null_info}\n")
    print(f"{'shape':<15}{str(cleaned_shape):>15}")
    print(f"{'null values':<15}{null_value:>15}\n")
    print(f"{'removed rows':<15}{removed_info['removed_rows']:>15}")
    print(f"{'removed columns':<15}{removed_info['removed_columns']:>15}\n")
    
    end_time = time.time()
    run_time = end_time - start_time
    print(f"{'run time':<15}{f'{run_time:.4f} seconds':>15}\n\n")
    return dataset
