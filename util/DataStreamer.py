import codecs, pprint, csv

class Example(object):

  all_tags = set()

  def __init__(self, data):
    self.data = data

  @classmethod
  def extract_code_sections(cls, mixed):
    # we avoid re because it is slow
    code = []
    noncode = []
    begins = mixed.split('<pre>')
    for match in begins:
      ends = match.split('</pre>')
      if len(ends) == 1:
        # this was the beginning before the first <pre>
        noncode += [ends[0]]
      else:
        code += [ends[0]]
        noncode += [ends[1]]
    return code, noncode

  @classmethod
  def from_csv_entries(cls, csv_entries, fields):
    data = {}
    for idx, f in enumerate(fields):
      data[f.lower()] = csv_entries[idx]
    data['tags'] = data['tags'].split()

    cls.all_tags = cls.all_tags.union(set(data['tags']))

    code, noncode = cls.extract_code_sections(data['body'])

    data['body'] = noncode
    data['code'] = code

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


