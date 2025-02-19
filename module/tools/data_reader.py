import pandas as pd
import os

def load_dataset(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    _, file_extension = os.path.splitext(file_path)
    
    try:
        if file_extension == '.csv':
            return pd.read_csv(file_path, low_memory=False, encoding_errors='ignore')
        elif file_extension in ['.xlsx', '.xls']:
            return pd.read_excel(file_path, engine='openpyxl' if file_extension == '.xlsx' else None)
        elif file_extension == '.json':
            return pd.read_json(file_path, precise_float=True)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
    except Exception as e:
        raise RuntimeError(f"Error loading file: {e}")
