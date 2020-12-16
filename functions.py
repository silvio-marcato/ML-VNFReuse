import numpy as np
import matplotlib.pyplot as py
import pandas as pd
import constant

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

def read_dataset(ds_path):
    df = pd.read_csv(ds_path)
    return df

def prune_dataset(df):
    grouped = df.groupby(df.context)
    descr = df.describe()
    max_context = descr['context']['max']
    i = 0
    final_df = pd.DataFrame(columns=constant.COLUMNS)
    while i <= max_context:
        contexts = grouped.get_group(i)
        best_beta_df = contexts[contexts.beta == contexts.beta.min()]
        best_ab_df = best_beta_df[best_beta_df.alpha == best_beta_df.alpha.min()]
        best_conf = best_ab_df.head(1)
        best_conf.drop(columns='context')
        final_df = pd.concat([final_df, best_conf], ignore_index=True)
        i += 1

    print('Dataset pruned!')

    return final_df

def save_pruned(df_to_save):
    df_to_save.to_csv('results/pruned_results.csv', mode='a', columns=constant.COLUMNS_PRUNED, index=False, header=True)

def train_model(pruned_df_path):
    pruned_df = read_dataset(pruned_df_path)
    pruned_df.describe()
    y = pruned_df['bin_conf']
    y = y.astype('int')
    x = pruned_df.drop(columns=['n_vnf','n_vcpu','alpha','beta','bin_conf'])
    clf = DecisionTreeClassifier(random_state=0)
    results = cross_val_score(clf, x, y, cv=2, scoring='accuracy')
    print("Accuracy: "+str(results.mean()))


