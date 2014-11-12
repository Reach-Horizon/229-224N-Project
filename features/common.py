import numpy as np
import cPickle as pickle
from scipy.sparse import csr_matrix
from bs4 import BeautifulSoup

def save_sparse_csr(filename,array):
  with open(filename + '.pickle', 'wb') as f:
    pickle.dump(array, f, protocol=pickle.HIGHEST_PROTOCOL)
    # np.savez(filename,data = array.data ,indices=array.indices,
    #          indptr =array.indptr, shape=array.shape )

def load_sparse_csr(filename):
  with open(filename + '.pickle', 'rb') as f:
    mat = pickle.load(f)
  return mat
  # loader = np.load(filename)
  # return csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
  #                      shape = loader['shape'])


def extract_code_sections(mixed):
  """
  splits a mixed text into a list of noncode sections and a list of code sections
  """
  noncode = mixed

  soup = BeautifulSoup(mixed)
  code = []
  for e in soup.find_all('pre'):
    if e.text:
      code += [e.text]
    noncode = noncode.replace(str(e), '')

  noncode = BeautifulSoup(noncode).text

  return code, noncode