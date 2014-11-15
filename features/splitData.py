"""Module responsible for taking given feature and label files and
splitting into respective train/test sets. These sets are written to disk."""

import common
from sklearn.cross_validation import train_test_split

matX = common.load_sparse_csr('100k.X')
matY = common.load_sparse_csr('100k.Y')

matXtrain, matYtrain, matXtest, matYtest = train_test_split(matX, matY, test_size = 0.2)

common.save_sparse_csr('100k_Xtrain', matXtrain)
common.save_sparse_csr('100k_Ytrain', matYtrain)
common.save_sparse_csr('100k_Xtest', matXtest)
common.save_sparse_csr('100k_Ytest', matYtest)
