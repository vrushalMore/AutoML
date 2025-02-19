import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize

def outlier_detection(df, limits=(0.05, 0.05)):
    df_winsorized = df.copy()
    
    for col in df.select_dtypes(include=[np.number]).columns:
        q1, q3 = np.percentile(df[col].dropna(), [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        if ((df[col] < lower_bound) | (df[col] > upper_bound)).any():
            df_winsorized[col] = winsorize(df[col], limits=limits)
    
    return df_winsorized
