"""Tests the ExternalMemory class implementation."""

import tensorflow as tf
import unittest

from .. dnc.external_memory import *

def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(ExternalMemoryTest))
    return suite

class ExternalMemoryTest(unittest.TestCase):
    
    def test_construction(self):
        graph = tf.Graph()
        with graph.as_default():
            with tf.Session(graph=graph) as sess:
                memory = ExternalMemory()
                self.assertIsInstance(memory, ExternalMemory)
                
    def test_cosine_similarity(self):
        graph = tf.Graph()
        with graph.as_default():
            with tf.Session(graph=graph) as sess:
                memory = ExternalMemory()
                diff = 1e-3 # take into account floating precision
                tests = [
                    { # simple example
                        'a': tf.constant([[1, 1, 0]], dtype=tf.float32),
                        'b': tf.constant([[1, 0, 1]], dtype=tf.float32),
                        'expected': [0.5],
                    }, { # divide by zero
                        'a': tf.constant([[0, 0, 0]], dtype=tf.float32),
                        'b': tf.constant([[1, 1, 1]], dtype=tf.float32),
                        'expected': [0],
                    }, { # equal vectors
                        'a': tf.constant([[100, 100, 100, 100]], dtype=tf.float32),
                        'b': tf.constant([[100, 100, 100, 100]], dtype=tf.float32),
                        'expected': [1],
                    }, { # batch size > 1
                        'a': tf.constant([
                            [1, 2, 3, 4, 5],
                            [120, 2, 40.5, 1, 1],
                        ], dtype=tf.float32),
                        'b': tf.constant([
                            [6, 7, 8, 9, 10],
                            [3, 60, 111, 1, 1],
                        ], dtype=tf.float32),
                        'expected': [0.9650, 0.3113],
                    }, { # cosine similarity produces a 1 for any 1-lengthed vector
                         # because the product and norm products are equal.
                        'a': tf.constant([[1], [2], [3], [4], [5], [6], [7]],
                                         dtype=tf.float32),
                        'b': tf.constant([[4], [4], [4], [4], [4], [4], [4]],
                                         dtype=tf.float32),
                        'expected': [1, 1, 1, 1, 1, 1, 1],
                    },
                ]
                for test in tests:
                    a = test['a']
                    b = test['b']
                    expected = test['expected']
                    similarities = sess.run(memory.cosine_similarity(a, b))
                    self.assertEqual(a.shape, b.shape)
                    self.assertEqual(similarities.shape[0], sess.run(a).shape[0])
                    for want, actual in zip(expected, similarities):
                        self.assertTrue(abs(want - actual) <= diff,
                                   msg="Test {}: expected {}, got {}".format(\
                                    tests.index(test), want, actual))
                        
    def test_content_based_read(self):
        graph = tf.Graph()
        with graph.as_default():
            with tf.Session(graph=graph) as sess:
                diff = 1e-4 # take into account floating precision
                tests = [
                    { # equal content addressed probabilities
                        'memory': [[1, 0, 1], [1, 0, 1], [1, 0, 1]],
                        'read_keys': [1, 1, 1],
                        'read_strengths': [1, 1, 1],
                        'equal_indices': [[0, 1, 2]],
                        'indices_sorted': [],
                    }, { # more similar slots of memory have greater values in c_t
                        'memory': [[1, 1, 1], [1, 0, 1], [0, 0, 1]],
                        'read_keys': [1, 1, 1],
                        'read_strengths': [1, 1, 1],
                        'equal_indices': [],
                        'indices_sorted': [[0, 1, 2]],
                    }, { # unequal dimensions of memory for NxM
                        'memory': [[-5, -4], [-3, -2], [-1, 0], [1, 2], [3, 4]],
                        'read_keys': [-1, 0],
                        'read_strengths': [1, 1, 1, 1, 1],
                        'equal_indices': [],
                        'indices_sorted': [[2, 3, 4], [2, 1, 0]],
                    }, { # independent slots are equal in probability
                        'memory': [[1, 2], [50, 60], [1, 2], [-30, 70], [1, 2], [120, 85]],
                        'read_keys': [1, 2],
                        'read_strengths': [1, 1, 1, 1, 1, 1],
                        'equal_indices': [[0, 2, 4]],
                        'indices_sorted': [],
                    }, { # read strengths heighten output with greater strengths
                        'memory': [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
                        'read_keys': [1, 1, 1],
                        'read_strengths': [1, 2, 3, 4],
                        'equal_indices': [],
                        'indices_sorted': [[3, 2, 1, 0]],
                    },
                ]
                for test in tests:
                    memory = ExternalMemory(memory_size=len(test['memory']),
                                            word_size=len(test['memory'][0]))
                    prev_state = ExternalMemoryState(
                        memory=tf.constant(test['memory'], dtype=tf.float32))
                    read_keys = tf.constant(test['read_keys'], dtype=tf.float32)
                    read_strengths = tf.constant(test['read_strengths'], dtype=tf.float32)
                    c_t_op, next_state = memory(read_keys, read_strengths, prev_state)
                    c_t = sess.run(c_t_op)
                    for equal_indices in test['equal_indices']:
                        for i in range(len(equal_indices) - 1):
                            index1 = equal_indices[i]
                            index2 = equal_indices[i+1]
                            self.assertTrue(abs(c_t[index1] - c_t[index2]) <= diff,
                                            msg="Test {}: expected index {} and {} to be equal "\
                                                "but were {} and {}".format(tests.index(test),
                                                                            index1,
                                                                            index2,
                                                                            c_t[index1],
                                                                            c_t[index2]))
                    for sorted in test['indices_sorted']:
                        for i in range(len(sorted) - 1):
                            index1 = sorted[i]
                            index2 = sorted[i+1]
                            self.assertTrue(c_t[index1] > c_t[index2],
                                            msg="Test {}: c_t[{}]={} not > c_t[{}]={}".format(
                                                tests.index(test), index1, c_t[index1],
                                                index2, c_t[index2]))

if __name__ == '__main__':
    unittest.main(verbosity=2)
