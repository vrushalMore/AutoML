import warnings
import time 

def summary(df):
    start_time = time.time()
    warnings.filterwarnings("ignore")
    feature_types = {
        "Boolean": df.select_dtypes(include=['bool']).shape[1],
        "Categorical": df.select_dtypes(include=['object', 'category']).shape[1],
        "Numeric": df.select_dtypes(include=['number']).shape[1]
    }
    print("data summary :\n")
    print("number of features:")
    for feature_type, count in feature_types.items():
        print(f"{feature_type: <15}{count}")
    
    num_examples = len(df)
    print("\n")
    print(f"{'training examples:':<15}{num_examples:>15}")
    end_time = time.time()
    run_time = end_time - start_time
    print("\n")
    print(f"{'run time':<15}{f'{run_time:.4f} seconds':>15}")
    print("\n\n")

