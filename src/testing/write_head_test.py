"""Tests the WriteHead class implementation."""

import tensorflow as tf
import numpy as np
import unittest

from .. dnc.write_head import WriteHead

def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(WriteHeadTest))
    return suite

class WriteHeadTest(unittest.TestCase):
    
    def test_construction(self):
        memory_size = 64
        word_size = 16
        graph = tf.Graph()
        with graph.as_default():
            with tf.Session(graph=graph) as session:
                write_head = WriteHead(memory_size, word_size)
                
                self.assertTrue(isinstance(write_head, WriteHead))
                self.assertEqual(write_head.memory_size, memory_size)
                self.assertEqual(write_head.word_size, word_size)
                
if __name__ == '__main__':
    unittest.main(verbosity=2)
