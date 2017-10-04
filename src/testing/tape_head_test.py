"""Tests the TapeHead class implementation."""

import tensorflow as tf
import unittest

from .. dnc.tape_head import TapeHead

def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TapeHeadTest))
    return suite

class TapeHeadTest(unittest.TestCase):
    
    def test_construction(self):
        graph = tf.Graph()
        with graph.as_default():
            with tf.Session(graph=graph) as session:
                tape_head = TapeHead()
                self.assertTrue(isinstance(tape_head, TapeHead))
                
if __name__ == '__main__':
    unittest.main(verbosity=2)
