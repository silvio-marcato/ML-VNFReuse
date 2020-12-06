import numpy as np
import matplotlib.pyplot as py
import pandas as pd

def read_dataset(ds_path):
    df = pd.read_csv(ds_path)
    return df

def prune_dataset(df):
    duplicates = df.duplicated(subset=['context'])
    grouped = df.groupby(df.context)
    context_0 = grouped.get_group(0)
    print(context_0)
    descr = df.describe()
    max_context = descr['context']['max']
    i = 0
    while i < max_context:
        contexts = grouped.get_group(i)
        print(contexts)
        i += 1



