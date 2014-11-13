import numpy as np
import json, bz2
from scipy.sparse import csr_matrix
from bs4 import BeautifulSoup

def save_sparse_csr(filename, array):
  out_file = bz2.BZ2File(filename + '.custom.sav.bz2', 'wb', compresslevel=9)
  out_file.write(str(array.shape[0]) + "," + str(array.shape[1]) + "\n")
  row_indices, col_indices = array.nonzero()
  for row, col in zip(row_indices.tolist(), col_indices.tolist()):
    out_file.write(str(row) + "," + str(col) + "\n")
  out_file.close()

def load_sparse_csr(filename):
  row_indices = []
  col_indices = []
  in_file = bz2.BZ2File(filename + '.custom.sav.bz2', 'rb', compresslevel=9)
  lines = in_file.readlines()
  in_file.close()

  shape = lines[0].strip("\n ").split(",")
  shape = tuple([int(n) for n in shape])

  for line in lines[1:]:
    if line:
      terms = line.strip("\n ").split(",")
      row_indices += [int(terms[0])]
      col_indices += [int(terms[1])]

  row_indices = np.array(row_indices)
  col_indices = np.array(col_indices)

  data = np.ones_like(row_indices, dtype=np.uint8)
  mat = csr_matrix((data, (row_indices, col_indices)), shape=shape, dtype=np.uint8)

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