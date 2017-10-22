"""Tests the TapeHead class implementation."""

import tensorflow as tf
import unittest

from numpy.testing import assert_array_almost_equal

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
                    'write_key': [[6, 7]],
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
                    'write_key': [[2], [1]],
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
