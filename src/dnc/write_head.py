"""A 'Write Head' for the DNC.

This is an implementation of a 'Write Head' used by the DNC
architecture as described in DeepMind's Nature paper:
http://www.nature.com/nature/journal/vaop/ncurrent/full/nature20101.html

Author: Austin Derrow-Pinion
"""

import collections
import numpy as np
import sonnet as snt
import tensorflow as tf

WriteHeadState = collections.namedtuple('WriteHeadState', (
    'memory', 'write_weights', 'linkage', 'usage'))

class WriteHead(snt.RNNCore):
    """Writes to external memory utilizing 'Temporal Linkage' and 'Usage'.
-------------------------------------------------------------------------------
    A normal computer simply writes to memory the exact value at an exact
    address in memory. This is differntiable and therefore trainable to
    write at useful locations in memory for a given task.
    
    Temporal Linkage is used to keep track of the order things are written in.
    This allows the write head to iterate forward and backwards through
    temporally related elements written in memory.
    
    Usage keeps track of what memory locations have and have not been used
    recently. If a memory location has not been used for a long time, the
    controller learns to free up this space to make it available for future
    writes to memory.
    """
    
    def __init__(self,
                 memory_size,
                 word_size,
                 name='write_head'):
        """Initializes a Write Head used in a DNC.
        
        Args:
            memory_size: The number of memory slots (N in the DNC paper).
            word_size: The width of each memory slot (W in the DNC paper).
            name: The name of the module (default 'write_head').
        """
        super(WriteHead, self).__init__(name=name)

        self.memory_size = memory_size
        self.word_size = word_size
        
        self._state_size = WriteHeadState(
            memory=tf.TensorShape([self.memory_size, self.word_size]),
            write_weights=tf.TensorShape([self.memory_size]),
            linkage=None,
            usage=None)
        
    def _build(self, inputs, prev_state):
        """Computes one timestep of computation for the Write Head.
        
        Args:
            inputs: A Tensor of shape [batch_size, input_size] emitted from
                the DNC controller. This hold data that controls what this
                write head does.
            prev_state: An instance of 'WriteHeadState' containing the
                previous state of this write head.
        Returns:
            An instance of 'WriteHeadState' representing the next state
            of this write head after computation finishes.
        """
        return prev_state
    
    @property
    def state_size(self):
        """Returns a description of the state size."""
        return self._state_size
    
    @property
    def output_size(self):
        """Returns the output shape."""
        return tf.TensorShape([0])
