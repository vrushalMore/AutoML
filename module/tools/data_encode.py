from sklearn.preprocessing import LabelEncoder

def encode_data(df):
    df_encoded = df.copy()
    
    for column in df_encoded.select_dtypes(include=['object']).columns:
        if df_encoded[column].nunique() == 1:
            df_encoded[column] = 0
        elif df_encoded[column].nunique() == len(df_encoded[column]):
            df_encoded[column] = df_encoded[column].astype('category').cat.codes
        else:
            df_encoded[column] = LabelEncoder().fit_transform(df_encoded[column].astype(str))
    
    return df_encoded
