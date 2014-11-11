import codecs, pprint, csv, json, bz2
from collections import Counter

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
    return cls.__init__(json.loads(json_str))

  @classmethod
  def extract_code_sections(cls, mixed):
    """
    splits a mixed text into a list of noncode sections and a list of code sections
    """
    # A class method is particular to its class (it can view class/static methods/variables)
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
    """
    returns an Example object corresponding to the csv_entries
    """

    data = {}
    for idx, f in enumerate(fields):
      data[f.lower()] = csv_entries[idx]
    data['tags'] = data['tags'].split()

    # don't do this automatically because it is very slow
    #code, noncode = cls.extract_code_sections(data['body'])

    #data['body'] = noncode
    #data['code'] = code

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
    with codecs.open(fname, 'rb', encoding='ascii', errors='ignore') as f:
      csv_f = csv.reader(f)
      fields = csv_f.next()

      for line in csv_f:
          example = Example.from_csv_entries(line, fields)
          yield example


