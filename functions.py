import numpy as np
import matplotlib.pyplot as py
import pandas as pd
import constant
import os

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler

def read_dataset(ds_path):
    df = pd.read_csv(ds_path)
    return df

def prune_dataset(df):
    grouped = df.groupby(df.context)
    descr = df.describe()
    max_context = descr['context']['max']
    #max_context = int(len(df.index/constant.BIN_NUMBER))
    i = 0
    final_df = pd.DataFrame(columns=constant.COLUMNS)
    while i <= max_context:
        contexts = grouped.get_group(i)
        best_beta_df = contexts[contexts.n_vcpu == contexts.n_vcpu.min()]
        best_ab_df = best_beta_df[best_beta_df.n_vnf == best_beta_df.n_vnf.min()]
        best_conf = best_ab_df.head(1)
        best_conf.drop(columns='context')
        final_df = pd.concat([final_df, best_conf], ignore_index=True)
        i += 1

    print('Dataset pruned!')

    return final_df

def save_pruned(df_to_save, header=False):
    df_to_save.to_csv('results/pruned/pruned_results.csv', mode='a', columns=constant.COLUMNS_PRUNED, index=False, header=header)

def train_model(pruned_df_path):
    pruned_df = read_dataset(pruned_df_path)
    pruned_df.describe()

    y = pruned_df['bin_conf'].values
    x = pruned_df.drop(columns=['rate','n_instances','n_instances_CA','n_instances_ST','n_instances_VS','n_vnf','n_vcpu','n_vnf','alpha','beta','bin_conf'])
    #x = pruned_df.drop(columns=['rate','bin_conf'])

    scaler = StandardScaler()
    x_normalize = scaler.fit_transform(x)

    svc_parameters = {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]}

    tree_parameters = {'criterion': ['gini', 'entropy'], 
                       'max_depth': range(1,10), 
                       'min_samples_split': range(2,10), 
                       'min_samples_leaf': range(1,5) 
                       }
    
    knn_parameters = {'n_neighbors': range (1,10),
                      'weights': ['uniform', 'distance']}
    
    clf = GridSearchCV(DecisionTreeClassifier(random_state=0), tree_parameters, scoring='accuracy')
    results = cross_val_score(clf, x_normalize, y, cv=5, scoring='accuracy')
    print("Accuracy Decision Tree: "+str(results.mean()))

    clf2 = GridSearchCV(SVC(), svc_parameters, scoring='accuracy')
    results2 = cross_val_score(clf2, x_normalize, y, cv=5, scoring ='accuracy')
    print("Accuracy SVC: "+str(results2.mean()))

    clf3 = LinearSVC(random_state=0)
    results3 = cross_val_score(clf3, x_normalize, y, cv=5, scoring='accuracy')
    print("Accuracy Linear SVC: "+str(results3.mean()))

    
    clf4 = GridSearchCV(KNeighborsClassifier(), knn_parameters, scoring='accuracy')
    results4 = cross_val_score(clf4, x_normalize, y, cv=5, scoring = 'accuracy')
    print("Accuracy KNN: "+str(results4.mean()))

    clf5 = RandomForestClassifier(random_state=0)
    results5 = cross_val_score(clf5, x_normalize, y, cv=5, scoring='accuracy')
    print("Accuracy Random Forest: "+str(results5.mean()))

def count_bins(pruned_df_path):
    pruned_df = read_dataset(pruned_df_path)
    total = pruned_df.count()
    bin_confs = pruned_df['bin_conf'].values
    unique, counts = np.unique(bin_confs, return_counts=True)
    occurrences = dict(zip(unique, counts))
    print(occurrences)
    print('bin0: %.2f' % (occurrences[0][1]/total*100))
    print('bin1: %.2f' % (occurrences[0][1]/total*100))
    print('bin2: %.2f' % (occurrences[0][1]/total*100))
    print('bin3: %.2f' % (occurrences[0][1]/total*100))

def merge_results():
    for i, entry in enumerate(os.listdir(constant.PATH)):
        fd = os.path.join(constant.PATH, entry)
        if os.path.isfile(fd):
            df = read_dataset(fd)
            pruned_df = prune_dataset(df)
            if i == 1:
                save_pruned(pruned_df, header=True)
            else:
                save_pruned(pruned_df)