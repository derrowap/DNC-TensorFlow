"""Tests the ExternalMemory class implementation."""

import tensorflow as tf
import unittest

from .. dnc import external_memory
from numpy.testing import assert_array_almost_equal


def suite():
    """Create testing suite for all tests in this module."""
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(ExternalMemoryTest))
    return suite


class ExternalMemoryTest(unittest.TestCase):
    """Tests for the ExternalMemory class."""

    def test_construction(self):
        """Test the construction of an ExternalMemory."""
        memory = external_memory.ExternalMemory()
        self.assertIsInstance(memory, external_memory.ExternalMemory)

    def test_content_weights(self):
        """Test content_weights function."""
        graph = tf.Graph()
        with graph.as_default():
            with tf.Session(graph=graph) as sess:
                diff = 1e-4  # take into account floating precision
                tests = [
                    {  # equal content addressed probabilities
                        'memory': [[[1, 0, 1], [1, 0, 1], [1, 0, 1]]],
                        'read_keys': [[[1, 1, 1]]],
                        'read_strengths': [[1]],
                        'equal_indices': [[[0, 1, 2]]],
                        'indices_sorted': [],
                    }, {  # similar slots of memory have greater values in c_t
                        'memory': [[[1, 1, 1], [1, 0, 1], [0, 0, 1]]],
                        'read_keys': [[[1, 1, 1]]],
                        'read_strengths': [[1]],
                        'equal_indices': [],
                        'indices_sorted': [[[0, 1, 2]]],
                    }, {  # unequal dimensions of memory for NxM
                        'memory': [[[-5, -4], [-3, -2], [-1, 0], [1, 2],
                                    [3, 4]]],
                        'read_keys': [[[-1, 0], [-1, 0]]],
                        'read_strengths': [[1]],
                        'equal_indices': [],
                        'indices_sorted': [[[2, 3, 4], [2, 1, 0]]],
                    }, {  # independent slots are equal in probability
                        'memory': [[[1, 2], [50, 60], [1, 2], [-30, 70],
                                    [1, 2], [120, 85]]],
                        'read_keys': [[[1, 2]]],
                        'read_strengths': [[1]],
                        'equal_indices': [[[0, 2, 4]]],
                        'indices_sorted': [],
                    }, {  # tests that num_reads > 1 works as expected
                        'memory': [[[1, 1, 1], [1, 1, 1],
                                    [1, 1, 1], [1, 1, 1]]],
                        'read_keys': [[[1, 1, 1], [1, 1, 1], [1, 1, 1]]],
                        'read_strengths': [[1, 1, 1]],
                        'equal_indices': [[[0, 1, 2], [0, 1, 2], [0, 1, 2]]],
                        'indices_sorted': [],
                    }, {  # test batch sizes work as expected
                        'memory': [[[1, 0, 1], [1, 0, 1], [1, 0, 1]],
                                   [[1, 1, 1], [1, 0, 1], [0, 0, 1]],
                                   [[1, 1, 1], [1, 0, 1], [0, 0, 1]]],
                        'read_keys': [[[1, 1, 1]], [[1, 1, 1]], [[0, 0, 1]]],
                        'read_strengths': [[1], [1], [1]],
                        'equal_indices': [[[0, 1, 2]], [], []],
                        'indices_sorted': [[], [[0, 1, 2]], [[2, 1, 0]]],
                    },
                ]
                for test in tests:
                    mem = external_memory.ExternalMemory(
                        memory_size=len(test['memory'][0]),
                        word_size=len(test['memory'][0][0]))
                    memory = tf.constant(test['memory'], dtype=tf.float32)
                    read_keys = tf.constant(test['read_keys'],
                                            dtype=tf.float32)
                    read_strengths = tf.constant(test['read_strengths'],
                                                 dtype=tf.float32)
                    c_t_op = mem.content_weights(read_keys, read_strengths,
                                                 memory)
                    c_t = sess.run(c_t_op)
                    batch_num = 0
                    for equal_indices_batch in test['equal_indices']:
                        read_num = 0
                        for equal_indices in equal_indices_batch:
                            for i in range(len(equal_indices) - 1):
                                index1 = equal_indices[i]
                                index2 = equal_indices[i + 1]
                                self.assertTrue(
                                    abs(c_t[batch_num][read_num][index1] -
                                        c_t[batch_num][read_num][index2]) <=
                                    diff,
                                    msg="Test {}: batch {} -> expected index"
                                        " {} and {} to be equal but were {} "
                                        "and {}".format(
                                            tests.index(test), batch_num,
                                            index1, index2,
                                            c_t[batch_num][read_num][index1],
                                            c_t[batch_num][read_num][index2]))
                            read_num += 1
                        batch_num += 1
                    batch_num = 0
                    for sorted_batch in test['indices_sorted']:
                        read_num = 0
                        for sorted in sorted_batch:
                            for i in range(len(sorted) - 1):
                                index1 = sorted[i]
                                index2 = sorted[i + 1]
                                self.assertTrue(
                                    c_t[batch_num][read_num][index1] >
                                    c_t[batch_num][read_num][index2],
                                    msg="Test {}: batch {} -> c_t[{}]={} not >"
                                        " c_t[{}]={}".format(
                                            tests.index(test), batch_num,
                                            index1,
                                            c_t[batch_num][read_num][index1],
                                            index2,
                                            c_t[batch_num][read_num][index2]))
                            read_num += 1
                        batch_num += 1

    def test_content_weights_read_strenghts(self):
        """Test content_weights function for effective read_strenghts."""
        graph = tf.Graph()
        with graph.as_default():
            with tf.Session(graph=graph) as sess:
                diff = 1e-4  # take into account floating precision
                tests = [
                    {  # tests that num_reads > 1 works as expected
                        'memory': [[[1, 1, 1], [1, 1, 1],
                                    [1, 1, 1], [1, 1, 1]]],
                        'read_keys': [[[1, 1, 1], [1, 1, 1], [1, 1, 1]]],
                        'read_strengths': [[1, 2, 3]],
                        'indices_sorted': [[0, 1, 2]],
                    }, {
                        'memory': [[[1, 1], [1, 1]], [[1, 1], [1, 1]]],
                        'read_keys': [[[1, 1], [1, 1]], [[1, 1], [1, 1]]],
                        'read_strengths': [[1, 2], [2, 1]],
                        'indices_sorted': [[0, 1], [1, 0]],
                    },
                ]
                for test in tests:
                    mem = external_memory.ExternalMemory(
                        memory_size=len(test['memory'][0]),
                        word_size=len(test['memory'][0][0]))
                    memory = tf.constant(test['memory'], dtype=tf.float32)
                    read_keys = tf.constant(test['read_keys'],
                                            dtype=tf.float32)
                    read_strengths = tf.constant(test['read_strengths'],
                                                 dtype=tf.float32)
                    c_t_op = mem.content_weights(read_keys, read_strengths,
                                                 memory)
                    c_t = sess.run(c_t_op)
                    batch_num = 0
                    for sorted_reads in test['indices_sorted']:
                        for i in range(len(sorted_reads) - 1):
                            index1 = sorted_reads[i]
                            index2 = sorted_reads[i + 1]
                            for j in range(len(c_t[batch_num][index1])):
                                self.assertTrue(
                                    abs(c_t[batch_num][index1][j] -
                                        c_t[batch_num][index2][j]) <= diff,
                                    msg="Test {}: expected c_t[{}][{}][{}] = "
                                        "{} == c_t[{}][{}][{}] = {}".format(
                                            tests.index(test),
                                            batch_num, index1, j,
                                            c_t[batch_num][index1][j],
                                            batch_num, index2, j,
                                            c_t[batch_num][index2][j]))
                        batch_num += 1

    def test_write_operation(self):
        """Test writing in external memory."""
        graph = tf.Graph()
        with graph.as_default():
            with tf.Session(graph=graph) as sess:
                tests = [
                    {  # do not erase or add memory, stay the same
                        'memory': [[[1, 2, 3], [4, 5, 6]],
                                   [[8, 9, 10], [11, 12, 13]]],
                        'write_weightings': [[0, 0], [0, 0]],
                        'erase_vector': [[0, 0, 0], [0, 0, 0]],
                        'write_vector': [[0, 0, 0], [0, 0, 0]],
                        'expected': [[[1, 2, 3], [4, 5, 6]],
                                     [[8, 9, 10], [11, 12, 13]]],
                    }, {  # basic write operation
                        'memory': [[[0, 0, 0], [0, 0, 0], [0, 0, 0]]],
                        'write_weightings': [[1, 1, 1]],
                        'erase_vector': [[0, 0, 0]],
                        'write_vector': [[1, 1, 1]],
                        'expected': [[[1, 1, 1], [1, 1, 1], [1, 1, 1]]],
                    }, {  # tests multiple batches
                        'memory': [[[0, 0], [0, 0]],
                                   [[0, 0], [0, 0]],
                                   [[0, 0], [0, 0]]],
                        'write_weightings': [[1, 1], [1, 1], [1, 1]],
                        'erase_vector': [[0, 0], [0, 0], [0, 0]],
                        'write_vector': [[1, 2], [3, 4], [5, 6]],
                        'expected': [[[1, 2], [1, 2]],
                                     [[3, 4], [3, 4]],
                                     [[5, 6], [5, 6]]],
                    }, {  # write weightings heighten written values in add
                        'memory': [[[0, 0], [0, 0]]],
                        'write_weightings': [[1, 2]],
                        'erase_vector': [[0, 0]],
                        'write_vector': [[1, 1]],
                        'expected': [[[1, 1], [2, 2]]],
                    }, {  # erase vector erases correct positions
                        'memory': [[[1, 2], [3, 4]]],
                        'write_weightings': [[1, 1]],
                        'erase_vector': [[.5, 1]],  # 0.5 erases half the value
                        'write_vector': [[0, 0]],
                        'expected': [[[.5, 0], [1.5, 0]]],
                    }, {  # write weighting of 0 prevents erase to that slot
                        'memory': [[[1, 2], [3, 4]]],
                        'write_weightings': [[0, 1]],
                        'erase_vector': [[.5, 1]],
                        'write_vector': [[0, 0]],
                        'expected': [[[1, 2], [1.5, 0]]],
                    }, {  # write weighting of 0 prevents adding to that slot
                        'memory': [[[0, 0], [0, 0]]],
                        'write_weightings': [[1, 0]],
                        'erase_vector': [[0, 0]],
                        'write_vector': [[1, 1]],
                        'expected': [[[1, 1], [0, 0]]],
                    }, {  # write weighting heightens erase operation
                        'memory': [[[1, 1], [1, 1]]],
                        'write_weightings': [[.5, 1.5]],
                        'erase_vector': [[.2, .5]],
                        'write_vector': [[0, 0]],
                        'expected': [[[.9, .75], [.7, .25]]],
                    }, {  # write and erase at same time
                        'memory': [[[1, 2], [3, 4]]],
                        'write_weightings': [[1, 1]],
                        'erase_vector': [[0.5, 1]],
                        'write_vector': [[2, 4]],
                        'expected': [[[2.5, 4], [3.5, 4]]],
                    },
                ]
                for test in tests:
                    mem = external_memory.ExternalMemory(
                        memory_size=len(test['memory'][0]),
                        word_size=len(test['memory'][0][0]))
                    write_weightings = tf.constant(
                        test['write_weightings'], dtype=tf.float32)
                    erase_vector = tf.constant(
                        test['erase_vector'], dtype=tf.float32)
                    write_vector = tf.constant(
                        test['write_vector'], dtype=tf.float32)
                    memory = tf.constant(
                        test['memory'], dtype=tf.float32)
                    next_memory_op = mem.write(write_weightings, erase_vector,
                                               write_vector, memory)
                    got = sess.run(next_memory_op)
                    assert_array_almost_equal(test['expected'], got)

if __name__ == '__main__':
    unittest.main(verbosity=2)
