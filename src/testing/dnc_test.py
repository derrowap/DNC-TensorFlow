"""Tests the DNC class implementation."""

import unittest

from .. dnc import dnc


def suite():
    """Create testing suite for all tests in this module."""
    suite = unittest.TestSuite()
    suite.addTest(DNCTest('test_construction'))
    return suite


class DNCTest(unittest.TestCase):
    """Tests for the DNC class."""

    def test_construction(self):
        """Test the construction of a DNC vector."""
        controller_config = {
            "hidden_size": 64,
        }
        memory_config = {
            "memory_size": 16,
            "word_size": 16,
            "num_reads": 4,
            "num_writes": 1,
        }
        output_size = 10
        d = dnc.DNC(controller_config, memory_config, output_size)
        self.assertIsInstance(d, dnc.DNC)

if __name__ == '__main__':
    unittest.main(verbosity=2)
