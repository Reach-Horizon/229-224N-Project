import os, sys

root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_dir)

from util.DataStreamer import Example

def assert_equals(expected, got):
    assert got == expected, 'expected %s but got %s instead' % (expected, got)

exampleA = Example({'title': 'Something Title C++ is the Ubuntu', 'body': 'I like C but not C++ <pre><code>#include "random_crap.h"</code></pre>', 'tags':['c++', 'c']})

exampleB = Example({'title': 'why does sklearn Suck so Much???', 'body': 'I used to like scikit-learn but then it cheated on me with C#. I hate C# !!!11!11!! Here I am talking about Microsoft and Mozilla \
and all I can say is when will Matlab release a new movie.<pre>from sys import os</pre>', 'tags':['python', 'c#']})

examples = [exampleA, exampleB]

mapping = [u'c#', u'java', u'php', u'javascript', u'android', u'jquery', u'c++', u'iphone', u'python', u'asp.net', 'c']