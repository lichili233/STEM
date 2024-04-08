import pandas as pd
from sklearn.model_selection import train_test_split

# %%
data = pd.read_csv('./raw_data/mtl_product.csv', index_col=False)

# %%
train_valid_set, test_set = train_test_split(data, test_size=0.1, random_state=42)
train_set, valid_set = train_test_split(train_valid_set, test_size=0.1, random_state=42)

# %%
train_set.to_csv('./train.csv', index=None)
valid_set.to_csv('./valid.csv', index=None)
test_set.to_csv('./test.csv', index=None)