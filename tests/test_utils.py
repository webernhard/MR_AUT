import os
import sys
import tempfile

import importlib.util

# load MR_AUT_utils.py directly so tests don't depend on package import behavior
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
mod_path = os.path.join(REPO_ROOT, 'MR_AUT_utils.py')
spec = importlib.util.spec_from_file_location('MR_AUT_utils', mod_path)
mr_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mr_utils)
read_items = mr_utils.read_items
sanitize_filename = mr_utils.sanitize_filename


def test_sanitize_filename_basic():
    s = 'This is a: test/item? name'
    out = sanitize_filename(s)
    assert ' ' not in out
    assert len(out) <= 80


def test_read_items_tmpfile():
    data = 'MR_AUTitem\napple\nbanana\n'
    with tempfile.NamedTemporaryFile('w+', delete=False, suffix='.csv') as f:
        f.write(data)
        fname = f.name
    try:
        items = read_items(fname)
        assert items == ['apple', 'banana']
    finally:
        os.unlink(fname)
