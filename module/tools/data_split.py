def split_dataset(data, target_column): 
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")
    
    X = data.drop(columns=[target_column])
    y = data[target_column]

    if X.empty:
        raise ValueError("Feature matrix is empty after dropping the target column.")
    
    return X, y
