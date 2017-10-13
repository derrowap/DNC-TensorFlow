"""Tests the TemporalLinkage class implementation."""

import tensorflow as tf
import unittest

from .. dnc import temporal_linkage
from numpy.testing import assert_array_almost_equal


def suite():
    """Create testing suite for all tests in this module."""
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TemporalLinkageTest))
    return suite


class TemporalLinkageTest(unittest.TestCase):
    """Tests for the TemporalLinkage class."""

    def test_construction(self):
        """Test the construction of a TemporalLinkage."""
        linkage = temporal_linkage.TemporalLinkage()
        self.assertIsInstance(linkage, temporal_linkage.TemporalLinkage)

    def test_updated_precedence_weights(self):
        """Test the updated_precedence_weights method in TemporalLinkage."""
        graph = tf.Graph()
        with graph.as_default():
            with tf.Session(graph=graph) as sess:
                tests = [
                    {  # unchanged precedence weights
                        'write_weightings': [[0, 0, 0]],
                        'precedence': [[0, 0, 0]],
                        'expected': [[0, 0, 0]],
                    }, {  # simple update case
                        'write_weightings': [[0.25, 0.25, 0.25, 0.25]],
                        'precedence': [[0, 0, 0, 0]],
                        'expected': [[0.25, 0.25, 0.25, 0.25]]
                    }, {  # full write_weightings fully replaces precedence
                        'write_weightings': [[0.25, 0.25, 0.25, 0.25]],
                        'precedence': [[0.8, 0.05, 0.15, 0]],
                        'expected': [[0.25, 0.25, 0.25, 0.25]],
                    }, {  # partial write_weightings adds to precedence
                        'write_weightings': [[0.25, 0.25]],
                        'precedence': [[0.2, 0.8]],
                        'expected': [[0.35, 0.65]],
                    }, {  # 0 write_weightings doesn't effect precedence
                        'write_weightings': [[0, 0, 0, 0]],
                        'precedence': [[0.4, 0.1, 0.2, 0.3]],
                        'expected': [[0.4, 0.1, 0.2, 0.3]],
                    }, {  # batch_size > 1 works as expected, independently
                        'write_weightings': [[0, 1], [0.25, 0.25], [0.5, 0.5]],
                        'precedence': [[0.5, 0.5], [0.3, 0.7], [0.2, 0.8]],
                        'expected': [[0, 1], [0.4, 0.6], [0.5, 0.5]],
                    },
                ]
                for test in tests:
                    w_t = tf.constant(
                        test['write_weightings'], dtype=tf.float32)
                    p_t = tf.constant(test['precedence'], dtype=tf.float32)
                    expected = test['expected']
                    linkage = temporal_linkage.TemporalLinkage(
                        memory_size=len(test['expected'][0]))
                    got_op = linkage.updated_precedence_weights(w_t, p_t)
                    got = sess.run(got_op)
                    assert_array_almost_equal(expected, got)

if __name__ == '__main__':
    unittest.main(verbosity=2)
