def split_dataset(data, target_column):
    x = data.drop(columns=[target_column])
    y = data[target_column]
    return x, y

