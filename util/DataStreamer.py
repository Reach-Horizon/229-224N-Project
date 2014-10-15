import codecs, pprint, csv

class Tag(object):
  def __init__(self):
    pass

class Example(object):
  def __init__(self, data):
    self.data = data

  @classmethod
  def from_csv_entries(cls, csv_entries, fields):
    data = {}
    for idx, f in enumerate(fields):
      data[f.lower()] = csv_entries[idx]
    data['tags'] = data['tags'].split()
    return Example(data)

  def __str__(self):
    return pprint.pformat(self.data)

class DataStreamer(object):

  @classmethod
  def load_from_file(cls, fname):
    with codecs.open(fname, 'rb', encoding='ascii', errors='ignore') as f:
      csv_f = csv.reader(f)
      fields = csv_f.next()

      for line in csv_f:
          example = Example.from_csv_entries(line, fields)
          yield example


