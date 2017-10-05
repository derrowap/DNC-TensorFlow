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
            with tf.Session(graph=graph) as session:
                memory = ExternalMemory()
                self.assertTrue(isinstance(memory, ExternalMemory))
                
if __name__ == '__main__':
    unittest.main(verbosity=2)
