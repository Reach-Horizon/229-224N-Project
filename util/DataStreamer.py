import codecs, pprint, csv, json, bz2

class Example(object):
  """
  An object representing a question and its tags
  """

  def __init__(self, data):
    """
    constructor
    """
    # this is a instance method
    # data is a instance variable
    self.data = data

  def to_json(self):
    return json.dumps(self.data)

  @classmethod
  def from_json(cls, json_str):
    return cls(json.loads(json_str))

  @classmethod
  def from_csv_entries(cls, csv_entries, fields):
    """
    returns an Example object corresponding to the csv_entries
    """

    data = {}
    for idx, f in enumerate(fields):
      data[f.lower()] = csv_entries[idx]
    data['tags'] = data['tags'].split()

    return Example(data)

  def __str__(self):
    """
    overrides the str representation of this object
    """
    return pprint.pformat(self.data)

class DataStreamer(object):
  """
  a library for parsing data files
  """

  @classmethod
  def load_from_bz2(cls, fname):
    infile = bz2.BZ2File(fname, 'rb', compresslevel=9)
    for line in infile:
        yield Example.from_json(line.strip("\n"))
    infile.close()

  @classmethod
  def load_from_file(cls, fname):
    """
    returns a generator over the Examples present in the file
    """

    f = bz2.BZ2File(fname, 'rb')
    #with codecs.open(fname, 'rb', encoding='ascii', errors='ignore') as f:
    csv_f = csv.reader(f)
    fields = csv_f.next()

    for line in csv_f:
        example = Example.from_csv_entries(line, fields)
        yield example
    f.close()


