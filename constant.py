#Constants

PATH = 'results/'
CURR_PATH = 'results/results18-03.csv'
PLOT_PATH = 'results_to_plot_3.csv'
PRUNED_PATH = 'results/pruned/pruned_results.csv'
PRUNED_PATH_TEST = 'results/pruned/pruned_results_test.csv'
COLUMNS_PRUNED = ['rate', 'n_requests', 'n_requests_CA', 'n_requests_ST', 'n_requests_VS', 'n_instances', 'n_instances_CA', 'n_instances_ST', 'n_instances_VS', 'n_vnf_req', 'n_vnf', 'n_vcpu', 'alpha', 'beta', 'bin_conf']
COLUMNS = ['context','rate', 'n_requests', 'n_requests_CA', 'n_requests_ST', 'n_requests_VS', 'n_instances', 'n_instances_CA', 'n_instances_ST', 'n_instances_VS', 'n_vnf_req', 'n_vnf', 'n_vcpu', 'alpha', 'beta', 'bin_conf']
BIN_NUMBER = 4
TRAIN_COLUMNS = ['n_instances_CA', 'n_instances_ST', 'n_instances_VS']
CLASS_LABEL = 'bin_conf'

SVC_PARAMS = {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4, 1e-5],
                     'C': [1,10,100,1000]}

TREE_PARAMS = {'criterion': ['entropy','gini'], 
                       'max_depth': range(1,10), 
                       'min_samples_split': range(2,10), 
                       'min_samples_leaf': range(1,5) 
                       }

KNN_PARAMS = {'n_neighbors': range (1,10),
                      'weights': ['uniform', 'distance']}

n_estimators = [100,200,300]
max_depth = range(1,10)
min_samples_split = range(2,10)
min_samples_leaf = range(1,5) 

RF_PARAMS = dict(n_estimators = n_estimators, max_depth = max_depth,  
              min_samples_split = min_samples_split, 
             min_samples_leaf = min_samples_leaf)