import numpy as np
import matplotlib.pyplot as py
import pandas as pd
import constant
import os
import joblib

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

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
    df_to_save.to_csv(constant.PRUNED_PATH_TEST, mode='a', columns=constant.COLUMNS_PRUNED, index=False, header=header)

def train_model(pruned_df_path):
    pruned_df = read_dataset(pruned_df_path)



    y_train = pruned_df['bin_conf'].values
    x_train = pruned_df.drop(columns=['rate','n_requests','n_requests_CA','n_requests_ST','n_requests_VS','n_instances','n_vnf_req','n_vcpu','n_vnf','alpha','beta','bin_conf'])
    #x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

    test_df = read_dataset(constant.PRUNED_PATH_TEST)
    x_test = test_df.drop(columns=['rate','n_requests','n_requests_CA','n_requests_ST','n_requests_VS','n_instances','n_vnf_req','n_vcpu','n_vnf','alpha','beta','bin_conf'])
    y_test = test_df['bin_conf'].values
    #x = pruned_df.drop(columns=['rate','bin_conf'])

    scaler = StandardScaler()
    x_normalize = scaler.fit_transform(x_train)
    x_normalize_test = scaler.fit_transform(x_test)

    svc_parameters = {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]}
    
    linear_svc_parameters = {'C': [1, 10, 100, 1000]}

    tree_parameters = {'criterion': ['gini', 'entropy'], 
                       'max_depth': range(1,10), 
                       'min_samples_split': range(2,10), 
                       'min_samples_leaf': range(1,5) 
                       }
    
    knn_parameters = {'n_neighbors': range (1,10),
                      'weights': ['uniform', 'distance']}

    n_estimators = [100,200]
    max_depth = [30,40]
    min_samples_split = [3,8]
    min_samples_leaf = [5,10] 

    hyperF = dict(n_estimators = n_estimators, max_depth = max_depth,  
              min_samples_split = min_samples_split, 
             min_samples_leaf = min_samples_leaf)
    

    clf3 = GridSearchCV(LinearSVC(random_state=0), linear_svc_parameters, cv=10, scoring='accuracy', n_jobs=-1)
    # results3 = cross_val_score(clf3, x_normalize, y_train, cv=20, scoring='accuracy')
    # print("Accuracy Linear SVC: "+str(results3.mean()))
    clf3.fit(x_normalize, y_train)
    y_pred = clf3.predict(x_normalize_test)
    print('Test accuracy: '+str(accuracy_score(y_test,y_pred)))

    clf5 = GridSearchCV(RandomForestClassifier(random_state=0), hyperF, cv=10, scoring='accuracy', n_jobs=-1)
    #clf5 = RandomForestClassifier(random_state=0)
    # results5 = cross_val_score(clf5, x_normalize, y_train, cv=10, scoring='accuracy')
    # print("Accuracy Random Forest: "+str(results5.mean()))
    clf5.fit(x_normalize, y_train)
    y_pred = clf5.predict(x_normalize_test)
    print('Test accuracy: '+str(accuracy_score(y_test,y_pred)))
    filename = 'rf_mode.sav'
    joblib.dump(clf5, filename)
    print(clf5.best_params_)

    clf = GridSearchCV(DecisionTreeClassifier(random_state=0), tree_parameters, cv=10, scoring='accuracy', n_jobs=-1)
    # results = cross_val_score(clf, x_normalize, y_train, cv=10, scoring='accuracy')
    # print("Accuracy Decision Tree: "+str(results.mean()))
    clf.fit(x_normalize, y_train)
    y_pred = clf.predict(x_normalize_test)
    print('Test accuracy: '+str(accuracy_score(y_test,y_pred)))
    filename = 'tree_mode.sav'
    joblib.dump(clf, filename)
    print(clf.best_params_)

    clf2 = GridSearchCV(SVC(), svc_parameters, cv=10, scoring='accuracy', n_jobs=-1)
    # results2 = cross_val_score(clf2, x_normalize, y_train, cv=10, scoring ='accuracy')
    # print("Accuracy SVC: "+str(results2.mean()))
    clf2.fit(x_normalize, y_train)
    y_pred = clf2.predict(x_normalize_test)
    print('Test accuracy: '+str(accuracy_score(y_test,y_pred)))
    filename2 = 'svc_mode.sav'
    joblib.dump(clf5, filename2)
    print(clf2.best_params_)

    clf4 = GridSearchCV(KNeighborsClassifier(), knn_parameters, cv=10, scoring='accuracy', n_jobs=-1)
    # results4 = cross_val_score(clf4, x_normalize, y_train, cv=10, scoring = 'accuracy')
    # print("Accuracy KNN: "+str(results4.mean()))
    clf4.fit(x_normalize, y_train)
    y_pred = clf4.predict(x_normalize_test)
    print('Test accuracy: '+str(accuracy_score(y_test,y_pred)))
    filename3 = 'knn_mode.sav'
    joblib.dump(clf4, filename3)
    print(clf4.best_params_)


def test_model(clf, x_train, y_train, x_test, y_test):
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    # target_names = ['1','2','3','4']
    # print(classification_report(y_test, y_pred, target_names=target_names))
    print('Test accuracy: '+str(accuracy_score(y_test,y_pred)))

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