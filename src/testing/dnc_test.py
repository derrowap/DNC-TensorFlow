"""Tests the DNC class implementation."""

import sonnet as snt
import tensorflow as tf
import unittest

from numpy.testing import assert_array_equal

from .. dnc import dnc


def suite():
    """Create testing suite for all tests in this module."""
    suite = unittest.TestSuite()
    suite.addTest(DNCTest('test_construction'))
    return suite


class DNCTest(unittest.TestCase):
    """Tests for the DNC class."""

    def test_construction(self):
        """Test the construction of a DNC."""
        output_size = 10
        d = dnc.DNC(output_size)
        self.assertIsInstance(d, dnc.DNC)

    def test_build(self):
        """Test the build of the DNC."""
        graph = tf.Graph()
        with graph.as_default():
            with tf.Session(graph=graph) as sess:
                output_size = 10
                memory_size = 20
                word_size = 8
                num_read_heads = 3
                hidden_size = 1

                tests = [{  # batch_size = 1
                    'input': [[1, 2, 3]],
                    'batch_size': 1
                }, {  # batch_size > 1
                    'input': [[1, 2, 3], [4, 5, 6]],
                    'batch_size': 2,
                }, {  # can handle 2D input with batch_size > 1
                    'input': [[[1, 2, 3],
                               [4, 5, 6],
                               [7, 8, 9]],
                              [[9, 8, 7],
                               [6, 5, 4],
                               [3, 2, 1]]],
                    'batch_size': 2,
                }, {  # 3D input with batch_size > 1
                    'input': [[[[1], [2]], [[3], [4]]],
                              [[[5], [6]], [[7], [8]]]],
                    'batch_size': 2,
                }]
                for test in tests:
                    i = tf.constant(test['input'], dtype=tf.float32)
                    batch_size = test['batch_size']
                    d = dnc.DNC(
                        output_size,
                        memory_size=memory_size,
                        word_size=word_size,
                        num_read_heads=num_read_heads,
                        hidden_size=hidden_size)
                    prev_state = d.initial_state(batch_size, dtype=tf.float32)
                    output_vector, dnc_state = d(i, prev_state)

                    assert_array_equal([batch_size, output_size],
                                       sess.run(tf.shape(output_vector)))
                    assert_array_equal(
                        [batch_size, num_read_heads, word_size],
                        sess.run(tf.shape(dnc_state.read_vectors)))

if __name__ == '__main__':
    unittest.main(verbosity=2)
