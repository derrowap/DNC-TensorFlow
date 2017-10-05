"""Tests the TemporalLinkage class implementation."""

import tensorflow as tf
import unittest

from .. dnc.temporal_linkage import TemporalLinkage

def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TemporalLinkageTest))
    return suite

class TemporalLinkageTest(unittest.TestCase):
    
    def test_construction(self):
        graph = tf.Graph()
        with graph.as_default():
            with tf.Session(graph=graph) as session:
                linkage = TemporalLinkage()
                self.assertTrue(isinstance(linkage, TemporalLinkage))
                
if __name__ == '__main__':
    unittest.main(verbosity=2)
