"""Tests the TapeHead class implementation."""

import tensorflow as tf
import unittest

from numpy.testing import assert_array_almost_equal, assert_array_equal

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

    def test_build(self):
        """Test the _build method of TapeHead."""
        graph = tf.Graph()
        with graph.as_default():
            with tf.Session(graph=graph) as sess:
                batch_size = 2
                num_read_heads = 3
                word_size = 4
                inputs = tf.zeros([
                    batch_size,
                    num_read_heads * word_size + 3 * word_size + 5 *
                    num_read_heads + 3
                ], dtype=tf.float32)
                tape = tape_head.TapeHead(word_size=word_size,
                                          num_read_heads=num_read_heads)
                prev_state = tape.initial_state(batch_size)
                read_vectors, next_state = tape(inputs, prev_state)
                read_vectors_shape = sess.run(tf.shape(read_vectors))
                assert_array_equal([batch_size, num_read_heads, word_size],
                                   read_vectors_shape)

                (
                    next_state_read_weights,
                    next_state_write_weights,
                    next_state_memory,
                    next_state_linkage,
                    next_state_usage,
                ) = next_state
                (
                    expected_read_weights_shape,
                    expected_write_weights_shape,
                    expected_memory_shape,
                    expected_linkage_shape,
                    expected_usage_shape,
                ) = tape.state_size

                assert_array_equal(
                    [batch_size] + expected_read_weights_shape.as_list(),
                    sess.run(tf.shape(next_state_read_weights)))
                assert_array_equal(
                    [batch_size] + expected_write_weights_shape.as_list(),
                    sess.run(tf.shape(next_state_write_weights)))
                assert_array_equal(
                    [batch_size] + expected_memory_shape.memory.as_list(),
                    sess.run(tf.shape(next_state_memory.memory)))
                assert_array_equal(
                    [batch_size] +
                    expected_linkage_shape.linkage_matrix.as_list(),
                    sess.run(tf.shape(next_state_linkage.linkage_matrix)))
                assert_array_equal(
                    [batch_size] +
                    expected_linkage_shape.precedence_weights.as_list(),
                    sess.run(tf.shape(next_state_linkage.precedence_weights)))
                assert_array_equal(
                    [batch_size] + expected_usage_shape.usage_vector.as_list(),
                    sess.run(tf.shape(next_state_usage.usage_vector)))

    def test_write_weighting(self):
        """Test the write_weighting method."""
        graph = tf.Graph()
        with graph.as_default():
            with tf.Session(graph=graph) as sess:
                tests = [{  # only allocation weighting
                    'write_gate': [[1]],
                    'allocation_gate': [[1]],
                    'allocation_weighting': [[0.5, 0.5]],
                    'write_content_weighting': [[0.5, 0.5]],
                    'expected': [[0.5, 0.5]],
                }, {  # allocation gate makes half of allocation_weighting and
                      # half of write_content_weighting
                    'write_gate': [[1]],
                    'allocation_gate': [[0.5]],
                    'allocation_weighting': [[0.2, 0.2]],
                    'write_content_weighting': [[0.5, 0.5]],
                    'expected': [[0.35, 0.35]],
                }, {  # write_gate halves the output
                    'write_gate': [[0.5]],
                    'allocation_gate': [[1]],
                    'allocation_weighting': [[0.5, 0.5]],
                    'write_content_weighting': [[0.5, 0.5]],
                    'expected': [[0.25, 0.25]],
                }, {  # batch_size > 1
                    'write_gate': [[0.5], [0.2]],
                    'allocation_gate': [[0.2], [0.5]],
                    'allocation_weighting': [[0.5, 0.5], [0.2, 0.8]],
                    'write_content_weighting': [[0.8, 0.2], [0.5, 0.5]],
                    'expected': [[0.37, 0.13], [0.07, 0.13]],
                }]
                for test in tests:
                    write_gate = tf.constant(test['write_gate'],
                                             dtype=tf.float32)
                    allocation_gate = tf.constant(test['allocation_gate'],
                                                  dtype=tf.float32)
                    allocation_weighting = tf.constant(
                        test['allocation_weighting'], dtype=tf.float32)
                    write_content_weighting = tf.constant(
                        test['write_content_weighting'], dtype=tf.float32)
                    expected = test['expected']
                    tape = tape_head.TapeHead(memory_size=len(expected[0]))
                    got = sess.run(tape.write_weighting(
                        write_gate, allocation_gate, allocation_weighting,
                        write_content_weighting))
                    assert_array_almost_equal(expected, got)

    def test_read_weights(self):
        """Test the read_weights method."""
        graph = tf.Graph()
        with graph.as_default():
            with tf.Session(graph=graph) as sess:
                tests = [{  # only backward values
                    'read_modes': [[[1, 0, 0]]],
                    'backward': [[[0.1, 0.2]]],
                    'content': [[[0.3, 0.4]]],
                    'forward': [[[0.5, 0.6]]],
                    'expected': [[[0.1, 0.2]]],
                }, {  # only content values
                    'read_modes': [[[0, 1, 0]]],
                    'backward': [[[0.1, 0.2]]],
                    'content': [[[0.3, 0.4]]],
                    'forward': [[[0.5, 0.6]]],
                    'expected': [[[0.3, 0.4]]],
                }, {  # only forward values
                    'read_modes': [[[0, 0, 1]]],
                    'backward': [[[0.1, 0.2]]],
                    'content': [[[0.3, 0.4]]],
                    'forward': [[[0.5, 0.6]]],
                    'expected': [[[0.5, 0.6]]],
                }, {  # mix of all 3 read modes
                    'read_modes': [[[0.2, 0.5, 0.3]]],
                    'backward': [[[0.1, 0.2]]],
                    'content': [[[0.3, 0.4]]],
                    'forward': [[[0.5, 0.6]]],
                    'expected': [[[0.32, 0.42]]],
                }, {  # num_read_heads > 1
                    'read_modes': [[[1, 0, 0], [0, 1, 0], [0, 0, 1]]],
                    'backward': [[[0.1, 0.2], [0.1, 0.2], [0.1, 0.2]]],
                    'content': [[[0.3, 0.4], [0.3, 0.4], [0.3, 0.4]]],
                    'forward': [[[0.5, 0.6], [0.5, 0.6], [0.5, 0.6]]],
                    'expected': [[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]],
                }, {  # batch_size > 1
                    'read_modes': [[[1, 0, 0]], [[0, 1, 0]], [[0, 0, 1]]],
                    'backward': [[[0.1, 0.2]], [[0.1, 0.2]], [[0.1, 0.2]]],
                    'content': [[[0.3, 0.4]], [[0.3, 0.4]], [[0.3, 0.4]]],
                    'forward': [[[0.5, 0.6]], [[0.5, 0.6]], [[0.5, 0.6]]],
                    'expected': [[[0.1, 0.2]], [[0.3, 0.4]], [[0.5, 0.6]]],
                }, {  # batch_size > 1 and num_read_heads > 1
                    'read_modes': [[[1, 0, 0], [1, 0, 0]],
                                   [[0, 1, 0], [0, 1, 0]],
                                   [[0, 0, 1], [0, 0, 1]]],
                    'backward': [[[0.1, 0.2], [0.1, 0.2]],
                                 [[0.1, 0.2], [0.1, 0.2]],
                                 [[0.1, 0.2], [0.1, 0.2]]],
                    'content': [[[0.3, 0.4], [0.3, 0.4]],
                                [[0.3, 0.4], [0.3, 0.4]],
                                [[0.3, 0.4], [0.3, 0.4]]],
                    'forward': [[[0.5, 0.6], [0.5, 0.6]],
                                [[0.5, 0.6], [0.5, 0.6]],
                                [[0.5, 0.6], [0.5, 0.6]]],
                    'expected': [[[0.1, 0.2], [0.1, 0.2]],
                                 [[0.3, 0.4], [0.3, 0.4]],
                                 [[0.5, 0.6], [0.5, 0.6]]],
                }]
                for test in tests:
                    read_modes = tf.constant(test['read_modes'],
                                             dtype=tf.float32)
                    b = tf.constant(test['backward'], dtype=tf.float32)
                    c = tf.constant(test['content'], dtype=tf.float32)
                    f = tf.constant(test['forward'], dtype=tf.float32)
                    expected = test['expected']
                    tape = tape_head.TapeHead(memory_size=len(expected[0][0]),
                                              num_read_heads=len(expected[0]))
                    got = sess.run(tape.read_weights(read_modes, b, c, f))
                    assert_array_almost_equal(expected, got)

    def test_interface_parameters(self):
        """Test the interface_parameters method."""
        graph = tf.Graph()
        with graph.as_default():
            with tf.Session(graph=graph) as sess:
                tests = [{
                    'N': 2,
                    'W': 2,
                    'R': 2,
                    'in': [[int(x) for x in range(23)]],
                    'read_keys': [[[0, 1], [2, 3]]],
                    'read_strengths': [[5.0181499279, 6.00671534]],
                    'write_key': [[[6, 7]]],
                    'write_strengths': [[9.000335406]],
                    'erase_vector': [[0.999876605424, 0.99995460]],
                    'write_vector': [[11, 12]],
                    'free_gates': [[0.9999977396757, 0.999999168]],
                    'allocation_gate': [[0.99999969]],
                    'write_gate': [[0.999999887]],
                    'read_modes': [[[0.090031, 0.244728, 0.665241],
                                    [0.090031, 0.244728, 0.665241]]],
                }, {  # batch_size > 1
                    'N': 1,
                    'W': 1,
                    'R': 1,
                    'in': [[float(x) for x in range(12)],
                           [float(x) / 2 for x in range(12)]],
                    'read_keys': [[[0.0]], [[0.0]]],
                    'read_strengths': [[2.313262], [1.974077]],
                    'write_key': [[[2]], [[1]]],
                    'write_strengths': [[4.048587], [2.701413]],
                    'erase_vector': [[0.982014], [0.880797]],
                    'write_vector': [[5], [2.5]],
                    'free_gates': [[0.997527], [0.952574]],
                    'allocation_gate': [[0.999089], [0.970688]],
                    'write_gate': [[0.999665], [0.982014]],
                    'read_modes': [[[0.090031, 0.244728, 0.665241]],
                                   [[0.186324, 0.307196, 0.506480]]],
                }]
                for test in tests:
                    th = tape_head.TapeHead(memory_size=test['N'],
                                            word_size=test['W'],
                                            num_read_heads=test['R'])
                    got = sess.run(th.interface_parameters(
                        tf.constant(test['in'], dtype=tf.float32)))
                    (
                        read_keys, read_strengths, write_key, write_strengths,
                        erase_vector, write_vector, free_gates,
                        allocation_gate, write_gate, read_modes
                    ) = got
                    assert_array_almost_equal(test['read_keys'], read_keys)
                    assert_array_almost_equal(test['read_strengths'],
                                              read_strengths)
                    assert_array_almost_equal(test['write_key'], write_key)
                    assert_array_almost_equal(test['write_strengths'],
                                              write_strengths)
                    assert_array_almost_equal(test['erase_vector'],
                                              erase_vector)
                    assert_array_almost_equal(test['write_vector'],
                                              write_vector)
                    assert_array_almost_equal(test['free_gates'], free_gates)
                    assert_array_almost_equal(test['allocation_gate'],
                                              allocation_gate)
                    assert_array_almost_equal(test['write_gate'], write_gate)
                    assert_array_almost_equal(test['read_modes'], read_modes)

if __name__ == '__main__':
    unittest.main(verbosity=2)
