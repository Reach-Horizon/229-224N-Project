import numpy as np
import json, bz2
from scipy.sparse import csr_matrix
from bs4 import BeautifulSoup

def save_sparse_csr(filename, array):
  d = {
    'indices': array.indices.tolist(),
    'indptr': array.indptr.tolist(),
    'shape': array.shape,
    }

  out_file = bz2.BZ2File(filename + '.custom.sav.bz2', 'wb', compresslevel=9)
  out_file.write(json.dumps(d))
  out_file.close()

def load_sparse_csr(filename):
  in_file = bz2.BZ2File(filename + '.custom.sav.bz2', 'rb', compresslevel=9)
  d = json.loads(in_file.read())
  in_file.close()

  data = np.ones_like(d['indices'], dtype=np.uint8)
  indices = d['indices']
  indptr = d['indptr']
  shape = d['shape']
  mat = csr_matrix((data, indices, indptr), shape=shape, dtype=np.uint8)

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