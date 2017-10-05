"""Tests the ExternalMemory class implementation."""

import tensorflow as tf
import unittest

from .. dnc.external_memory import ExternalMemory

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
        self.assertTrue(True)
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

if __name__ == '__main__':
    unittest.main(verbosity=2)
