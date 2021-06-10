import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import constant
import plots
import os
import joblib
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import classification_report, accuracy_score, plot_roc_curve



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

def read_dataset(ds_path):
    df = pd.read_csv(ds_path)
    return df

def train_model():

    #Read train set and apply rounding to average number of instances (we work with integers)
    pruned_df = read_dataset(constant.PRUNED_PATH)    
    x_train = pruned_df[constant.TRAIN_COLUMNS]
    y_train = pruned_df[constant.CLASS_LABEL].values

    for n_inst in x_train:
        x_train[n_inst] = x_train[n_inst].apply(np.around)
    
    #Read test set and apply rounding
    #x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size = 0.33, random_state=42)
    # test_df = read_dataset(constant.PRUNED_PATH_TEST)
    # x_test = test_df[constant.TRAIN_COLUMNS]
    # y_test = test_df[constant.CLASS_LABEL].values

    # for n_inst in x_test:
    #     x_test[n_inst] = x_test[n_inst].apply(np.around)
    
    #Oversampling
    # oversample = RandomOverSampler(sampling_strategy='minority')
    # x_train, y_train = oversample.fit_resample(x_train, y_train)

    #Normalizing
    # scaler = StandardScaler()
    # x_normalize = scaler.fit_transform(x_train)
    # x_normalize_test = scaler.transform(x_test) #not fit because we fit on the training
    x_normalize = x_train
    x_normalize_test = x_train
    y_test=y_train
 
    #Random Forest
    #clf5 = GridSearchCV(RandomForestClassifier(random_state=0), constant.RF_PARAMS, cv=10, scoring='accuracy', n_jobs=-1)
    clf5 = RandomForestClassifier(random_state=0, n_estimators=200, max_depth=9, min_samples_split=2, min_samples_leaf=1, n_jobs=-1)
    results5 = cross_val_score(clf5, x_normalize, y_train, cv=10, scoring='accuracy')
    print("Accuracy Random Forest: "+str(results5.mean()))
    clf5.fit(x_normalize, y_train)
    y_pred = clf5.predict(x_normalize_test)
    print('Test accuracy: '+str(accuracy_score(y_test,y_pred)))
    filename5 = 'rf_mode.sav'
    joblib.dump(clf5, filename5)
    print(clf5.best_params_)

    #Decision Tree
    #clf = GridSearchCV(DecisionTreeClassifier(random_state=0), constant.TREE_PARAMS, cv=10, scoring='accuracy', n_jobs=-1)
    clf = DecisionTreeClassifier(random_state=0, criterion='entropy', max_depth=5, min_samples_split=2, min_samples_leaf=1)
    results = cross_val_score(clf, x_normalize, y_train, cv=10, scoring='accuracy')
    print("Accuracy Decision Tree: "+str(results.mean()))
    clf.fit(x_normalize, y_train)
    y_pred = clf.predict(x_normalize_test)
    print('Test accuracy: '+str(accuracy_score(y_test,y_pred)))
    filename = 'tree_mode.sav'
    joblib.dump(clf, filename)
    print(clf.best_params_)
    #plots.my_plot_tree(clf)

    #Support Vector
    #clf2 = GridSearchCV(SVC(random_state=0), constant.SVC_PARAMS, cv=10, scoring='accuracy', n_jobs=-1)
    clf2 = SVC(random_state=0, C=100, gamma=0.001)
    results2 = cross_val_score(clf2, x_normalize, y_train, cv=10, scoring ='accuracy')
    print("Accuracy SVC: "+str(results2.mean()))
    clf2.fit(x_normalize, y_train)
    y_pred = clf2.predict(x_normalize_test)
    print('Test accuracy: '+str(accuracy_score(y_test,y_pred)))
    filename2 = 'svc_mode.sav'
    joblib.dump(clf2, filename2)
    print(clf2.best_params_)

    #K-Nearest Neighbors
    #clf4 = GridSearchCV(KNeighborsClassifier(), constant.KNN_PARAMS, cv=10, scoring='accuracy', n_jobs=-1)
    clf4 = KNeighborsClassifier(n_neighbors=3, weights='distance', n_jobs=-1)
    results4 = cross_val_score(clf4, x_normalize, y_train, cv=10, scoring = 'accuracy')
    print("Accuracy KNN: "+str(results4.mean()))
    clf4.fit(x_normalize, y_train)
    y_pred = clf4.predict(x_normalize_test)
    print('Test accuracy: '+str(accuracy_score(y_test,y_pred)))
    filename3 = 'knn_mode.sav'
    joblib.dump(clf4, filename3)
    print(clf4.best_params_)
    #plots.plot_clf(clf4, 'K-Nearest Neighbors Classifier', x_normalize, y_train)



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
    tot = sum(occurrences.values())
    for i, occ in enumerate(occurrences.values()):
        perc = occ/tot*100
        print(str(i)+' '+str(occ)+' '+str(perc))
    print(occurrences)
    print(tot)
    explode = (0, 0, 0, 0)
    colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']
    fig1, ax1 = plt.subplots()
    labels = ['Conf. 0', 'Conf. 1', 'Conf. 2', 'Conf. 3']
    ax1.pie(list(occurrences.values()), explode=explode, labels=labels, autopct='%1.1f%%', colors=colors,
        shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.show()

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