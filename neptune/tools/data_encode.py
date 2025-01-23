from sklearn.preprocessing import LabelEncoder

def encode_data(df):
    label_encoder = LabelEncoder()
    df_encoded = df.copy()
    
    for column in df_encoded.columns:
        if df_encoded[column].dtype == 'object':
            df_encoded[column] = label_encoder.fit_transform(df_encoded[column])
    
    return df_encoded
