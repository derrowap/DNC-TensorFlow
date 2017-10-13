"""Tests the Usage class implementation."""

import unittest

from .. dnc import usage


def suite():
    """Create testing suite for all tests in this module."""
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(UsageTest))
    return suite


class UsageTest(unittest.TestCase):
    """Tests for the Usage class."""

    def test_construction(self):
        """Test the construction of a Usage vector."""
        usage_vector = usage.Usage()
        self.assertIsInstance(usage_vector, usage.Usage)

if __name__ == '__main__':
    unittest.main(verbosity=2)
