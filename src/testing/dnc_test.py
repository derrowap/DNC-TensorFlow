"""Tests the DNC class implementation."""

import tensorflow as tf
import numpy as np
import unittest

from .. dnc.dnc import DNC

class DNCTest(unittest.TestCase):
    
    def test_construction(self):
        controller_config = {
            "hidden_size": 64,
        }
        memory_config = {
            "memory_size": 16,
            "word_size": 16,
            "num_reads": 4,
            "num_writes": 1,
        }
        output_size = 10
        graph = tf.Graph()
        with graph.as_default():
            with tf.Session(graph=graph) as session:
                dnc = DNC(controller_config, memory_config, output_size)
                
                self.assertTrue(isinstance(dnc, DNC))
                
if __name__ == '__main__':
    unittest.main(verbosity=2)
