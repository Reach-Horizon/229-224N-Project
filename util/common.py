import numpy as np
import logging, bz2
from scipy.sparse import csr_matrix
from bs4 import BeautifulSoup

def save_dense(filename, array):
  np.savez(filename + '.dense', array)

def load_dense(filename):
  return np.load(filename + '.dense' + '.npz')

def save_sparse_csr(filename, array):
  out_file = bz2.BZ2File(filename + '.custom.sav.bz2', 'wb', compresslevel=9)
  out_file.write(str(array.shape[0]) + "," + str(array.shape[1]) + "\n")
  row_indices, col_indices = array.nonzero()
  logging.info("saving (%s, %s) sparse matrix to %s" %(array.shape[0], array.shape[1], filename))
  for row, col in zip(row_indices.tolist(), col_indices.tolist()):
    out_file.write(str(row) + "," + str(col) + "," + str(array[row, col]) + "\n")
  out_file.close()

def load_sparse_csr(filename):
  row_indices = []
  col_indices = []
  data = []

  in_file = bz2.BZ2File(filename + '.custom.sav.bz2', 'rb', compresslevel=9)
  first_line = in_file.readline().strip(" \n")
  shape = tuple(first_line.split(","))

  for line in in_file:
    if line:
      terms = line.strip("\n ").split(",")
      row_indices += [int(terms[0])]
      col_indices += [int(terms[1])]
      data += [float(terms[2])]
  in_file.close()

  logging.info("loading (%s, %s) sparse matrix from %s" %(shape[0], shape[1], filename))

  row_indices = np.array(row_indices)
  col_indices = np.array(col_indices)
  data = np.array(data)

  mat = csr_matrix((data, (row_indices, col_indices)), shape=shape)

  return mat

import string
def extract_code_sections(mixed):
  """
  splits a mixed text into a list of noncode sections and a list of code sections
  """

  soup = BeautifulSoup(mixed)
  code = []
  for e in soup.find_all('pre'):
    e.extract()
    code += [e.text]
  for e in soup.find_all('code'):
    e.extract()
    code += [e.text]
  noncode = soup.text

  return "\n".join(code), noncode


def get_dataset_for_class(k, X, Y, fair_sampling=True, restrict_sample_size=0):
    # get the Ys corresponding to this class
    my_Y = np.squeeze(np.asarray(Y[:,k]))

    # get the negative and positive examples for this class
    pos_indices = np.where(my_Y == 1)[0]
    neg_indices = np.where(my_Y == 0)[0]

    # have too many negative examples, so subsample until we have equal number of negative and positive
    if fair_sampling:
        np.random.shuffle(neg_indices)
        neg_indices = neg_indices[:len(pos_indices)]

    if restrict_sample_size:
        if len(pos_indices) > restrict_sample_size:
            pos_indices = pos_indices[:restrict_sample_size]
        if len(neg_indices) > restrict_sample_size:
            neg_indices = neg_indices[:restrict_sample_size]

    # merge the training indices
    train_indices = np.hstack((pos_indices, neg_indices))
    np.random.shuffle(train_indices)

    # train the classifier for this class
    my_X = X[train_indices, :]
    my_Y = my_Y[train_indices]

    return my_X, my_Y