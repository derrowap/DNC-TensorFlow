"""Tests the TapeHead class implementation."""

import unittest

from .. dnc import tape_head


def suite():
    """Create testing suite for all tests in this module."""
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TapeHeadTest))
    return suite


class TapeHeadTest(unittest.TestCase):
    """Tests for the TapeHead class."""

    def test_construction(self):
        """Test the construction of a TapeHead vector."""
        tape = tape_head.TapeHead()
        self.assertIsInstance(tape, tape_head.TapeHead)

if __name__ == '__main__':
    unittest.main(verbosity=2)
