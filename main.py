import functions as f
import constant


df = f.read_dataset(constant.PATH)

pruned_df = f.prune_dataset(df)
f.save_pruned(pruned_df)


f.train_model(constant.PRUNED_PATH)
