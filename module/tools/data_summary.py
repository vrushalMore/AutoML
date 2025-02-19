import warnings
import time
import pandas as pd

def summary(df):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    
    start_time = time.time()
    warnings.filterwarnings("ignore")
    
    feature_types = {
        "Boolean": df.select_dtypes(include=['bool']).shape[1],
        "Categorical": df.select_dtypes(include=['object', 'category']).shape[1],
        "Numeric": df.select_dtypes(include=['number']).shape[1]
    }

    num_examples = len(df)
    
    print("\nData Summary:")
    print("=" * 30)
    print(f"{'Feature Type':<15} {'Count':>10}")
    print("-" * 30)
    for feature_type, count in feature_types.items():
        print(f"{feature_type:<15} {count:>10}")
    
    print("\n")
    print(f"{'Training Examples:':<20} {num_examples:>10}")

    run_time = time.time() - start_time
    print(f"{'Run Time:':<20} {run_time:.4f} seconds")
    print("=" * 30, "\n")

