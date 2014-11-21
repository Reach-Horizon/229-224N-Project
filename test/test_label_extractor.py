import os, sys

root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_dir)

from util.extract_labels import get_Y_from_examples
from test.scaffold import examples, mapping, assert_equals


Y = get_Y_from_examples(mapping, examples).todense()

assert_equals(1, Y[0, mapping.index('c++')])
assert_equals(1, Y[0, mapping.index('c')])

assert_equals(1, Y[1, mapping.index('python')])
assert_equals(1, Y[1, mapping.index('c#')])

print '<passed> test_label_extractor'

