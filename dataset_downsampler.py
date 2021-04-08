# loads and downsamples and pickles forest covertype data for user study

import pandas as pd
from lale.datasets import covtype_df
from sklearn.model_selection import train_test_split
from lale.lib.lale import categorical
import pickle

TRAIN_SIZE = 5000

(train_X_all, train_y_all), (test_X, test_y) = covtype_df(test_size=5000)
train_X, other_X, train_y, other_y = train_test_split(
    train_X_all, train_y_all, train_size=TRAIN_SIZE, stratify=train_y_all
)
constant_columns = categorical(max_values=1)(train_X)
train_X = train_X.drop(constant_columns, axis=1)
test_X = test_X.drop(constant_columns, axis=1)
#pd.options.display.max_columns = None
#pd.concat([train_y, train_X], axis=1)

with open('train_x.pickle', 'wb') as f:
    pickle.dump(train_X, f)

with open('test_x.pickle', 'wb') as f:
    pickle.dump(test_X, f)

with open('train_y.pickle', 'wb') as f:
    pickle.dump(train_y, f)

with open('test_y.pickle', 'wb') as f:
    pickle.dump(test_y, f)
