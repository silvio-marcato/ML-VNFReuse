import functions as f
import os
import constant
import joblib
import plots

# f.merge_results()

# df = f.read_dataset(constant.CURR_PATH)
# pruned_df = f.prune_dataset(df)
# f.save_pruned(pruned_df, header=True)


f.train_model(constant.PRUNED_PATH)

#plots.plot_histogram(constant.PLOT_PATH)

#f.count_bins(constant.PRUNED_PATH)
# rf = joblib.load('rf_mod-e.sav')
# print(rf.best_params_)
