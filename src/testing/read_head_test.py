"""Tests the ReadHead class implementation."""

import tensorflow as tf
import unittest

from .. dnc.read_head import ReadHead

def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(ReadHeadTest))
    return suite

class ReadHeadTest(unittest.TestCase):
    
    def test_construction(self):
        memory_size = 64
        word_size = 16
        read_mode = [0.1, 0.3, 0.6]
        graph = tf.Graph()
        with graph.as_default():
            with tf.Session(graph=graph) as session:
                read_head = ReadHead(memory_size, word_size, read_mode)
                
                self.assertTrue(isinstance(read_head, ReadHead))
                self.assertEqual(read_head.memory_size, memory_size)
                self.assertEqual(read_head.word_size, word_size)
                self.assertEqual(read_head.read_mode, read_mode)
                
if __name__ == '__main__':
    unittest.main(verbosity=2)
