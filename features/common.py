import numpy as np
import cPickle as pickle
from scipy.sparse import csr_matrix
from bs4 import BeautifulSoup

def save_sparse_csr(filename, array):
  # cannot use pickle or np.save because of python bug: http://bugs.python.org/issue11564
  np.savetxt(filename+'.gz', array)

def load_sparse_csr(filename):
  mat = np.loadtxt(filename+'.gz')
  return mat

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