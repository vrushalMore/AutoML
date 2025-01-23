from .tools.data_clean import clean
from .tools.model_eval import evaluation
from .tools.data_split import split_dataset
from .tools.data_encode import encode_data
import time 
from .tools.data_reader import load_dataset

def AutoML(data_string, target, type):
    dataset = load_dataset(data_string)
    start = time.time()
    dataset = clean(dataset)
    categorical_columns = dataset.select_dtypes(include=['object']).columns
    if len(categorical_columns) > 0:
        dataset = encode_data(dataset)
    x, y = split_dataset(dataset, target)
    evaluation(x, y, type)
    end = time.time()
    runtime = end - start
    print(f"Total runtime : {runtime:.4f} s")

