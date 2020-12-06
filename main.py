import functions as f
import constant


df = f.read_dataset(constant.PATH)

f.prune_dataset(df)
