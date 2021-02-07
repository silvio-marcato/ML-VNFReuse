import functions as f
import os
import constant

#f.merge_results()

# df = f.read_dataset(constant.CURR_PATH)
# pruned_df = f.prune_dataset(df)
# f.save_pruned(pruned_df)


f.train_model(constant.PRUNED_PATH)

#f.count_bins(constant.PRUNED_PATH)
