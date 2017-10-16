"""Tests the Usage class implementation."""

import tensorflow as tf
import unittest

from .. dnc import usage
from numpy.testing import assert_array_almost_equal, assert_array_equal


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

    def test_updated_usage_vector(self):
        """Test the updated_usage_vector method."""
        graph = tf.Graph()
        with graph.as_default():
            with tf.Session(graph=graph) as sess:
                tests = [{  # most basic example of all 0
                    'u_t': [[0, 0]],
                    'w_t': [[0, 0]],
                    'phi': [[0, 0]],
                    'expected': [[0, 0]],
                }, {  # phi = 0 wipes the usage vector
                    'u_t': [[0.5, 0.5]],
                    'w_t': [[0.5, 0.5]],
                    'phi': [[0, 0]],
                    'expected': [[0, 0]],
                }, {  # basic non-zero example
                    'u_t': [[0.5, 0.5]],
                    'w_t': [[0.5, 0.5]],
                    'phi': [[0.5, 0.5]],
                    'expected': [[.375, .375]],
                }, {  # u_t = 0 results in this just being w_t * phi
                    'u_t': [[0, 0, 0]],
                    'w_t': [[0.1, 0.2, 0.3]],
                    'phi': [[0.3, 0.2, 0.1]],
                    'expected': [[0.03, 0.04, 0.03]],
                }, {  # w_t = 0 results in this just being u_t * phi
                    'u_t': [[0.1, 0.2, 0.3]],
                    'w_t': [[0, 0, 0]],
                    'phi': [[0.3, 0.2, 0.1]],
                    'expected': [[0.03, 0.04, 0.03]],
                }, {  # batch_size > 0 works independently
                    'u_t': [[0.5, 0.5], [0, 0], [0.2, 0.3]],
                    'w_t': [[0.5, 0.5], [0.3, 0.4], [0.9, 0.1]],
                    'phi': [[0.5, 0.5], [0.4, 0.6], [0.3, 0.2]],
                    'expected': [[.375, .375], [.12, .24], [.276, .074]],
                }]
                for test in tests:
                    u_t = tf.constant(test['u_t'], dtype=tf.float32)
                    w_t = tf.constant(test['w_t'], dtype=tf.float32)
                    phi = tf.constant(test['phi'], dtype=tf.float32)
                    expected = test['expected']
                    u = usage.Usage(memory_size=len(expected[0]))
                    got = sess.run(u.updated_usage_vector(u_t, w_t, phi))
                    assert_array_almost_equal(expected, got)

    def test_memory_retention_vector(self):
        """Test the memory_retention_vector method."""
        graph = tf.Graph()
        with graph.as_default():
            with tf.Session(graph=graph) as sess:
                tests = [{  # no locations can be freed
                    'w_r': [[[0, 0]]],
                    'free_gates': [[0]],
                    'expected': [[1, 1]],
                }, {  # all locations can be freed
                    'w_r': [[[1, 1]]],
                    'free_gates': [[1]],
                    'expected': [[0, 0]],
                }, {  # partial memory retention
                    'w_r': [[[0.5, 0.5]]],
                    'free_gates': [[0.5]],
                    'expected': [[0.75, 0.75]],
                }, {  # num_reads > 1
                    'w_r': [[[1, 0], [0.5, 0.5], [0, 0]]],
                    'free_gates': [[1, 0.5, 0]],
                    'expected': [[0, 0.75]],
                }, {  # batch_size > 1
                    'w_r': [[[1, 0]], [[0.5, 0.5]], [[0, 0]]],
                    'free_gates': [[1], [0.5], [0]],
                    'expected': [[0, 1], [0.75, 0.75], [1, 1]],
                }, {  # batch_size > 1 and num_reads > 1
                    'w_r': [[[1, 0], [0.5, 0.5]], [[0.5, 0.5], [0, 0]]],
                    'free_gates': [[0.5, 0.2], [0.5, 0.8]],
                    'expected': [[.45, .9], [.75, .75]],
                }]
                for test in tests:
                    w_r = tf.constant(test['w_r'], dtype=tf.float32)
                    f_t = tf.constant(test['free_gates'], dtype=tf.float32)
                    expected = test['expected']
                    usage_vector = usage.Usage(memory_size=len(expected[0]))
                    got_op = usage_vector.memory_retention_vector(w_r, f_t)
                    got = sess.run(got_op)
                    assert_array_almost_equal(expected, got)

    def test_sorted_indices(self):
        """Test the sorted_indices method."""
        graph = tf.Graph()
        with graph.as_default():
            with tf.Session(graph=graph) as sess:
                tests = [{  # basic case
                    'usage': [[1, 2, 3]],
                    'expected': [[0, 1, 2]],
                }, {  # batch_size > 1
                    'usage': [[3, 6, 5, 1], [2, 9, 7, 3]],
                    'expected': [[3, 0, 2, 1], [0, 3, 2, 1]],
                }, {  # size of 1
                    'usage': [[1], [2], [3]],
                    'expected': [[0], [0], [0]],
                }]
                for test in tests:
                    usage_vector = tf.constant(test['usage'], dtype=tf.float32)
                    expected = test['expected']
                    u = usage.Usage(memory_size=len(expected[0]))
                    got = sess.run(u.sorted_indices(usage_vector))
                    assert_array_equal(expected, got)

if __name__ == '__main__':
    unittest.main(verbosity=2)
