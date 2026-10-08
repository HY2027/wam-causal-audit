import os
from pathlib import Path
import unittest
from unittest.mock import patch
from wam_causal_audit.paths import resolve


class PathTests(unittest.TestCase):
    def test_default_is_release(self):
        with patch.dict(os.environ, {}, clear=True):
            path=Path(resolve('@WORKSPACE@/FastWAM'))
            self.assertTrue(path.is_dir())
            self.assertTrue(str(path).endswith('reference/workspace/FastWAM'))

    def test_explicit_overrides(self):
        with patch.dict(os.environ, {'WAM_WORKSPACE':'/example/work','WAM_DATA':'/example/data'}):
            self.assertEqual(resolve('@DATA@/a'),'/example/data/a')
            self.assertEqual(resolve('@WORKSPACE@/b'),'/example/work/b')

    def test_nonpath_unchanged(self):
        self.assertEqual(resolve('A10 = current donor; future recipient'),
                         'A10 = current donor; future recipient')


if __name__=='__main__':unittest.main()
