"""DNC core implementation.

-------------------------------------------------------------------------------
This is an implementation of the complete Differentiable Neural Computer
architecture as described in DeepMind's Nature paper:
http://www.nature.com/nature/journal/vaop/ncurrent/full/nature20101.html

Author: Austin Derrow-Pinion
"""

import numpy as np
import sonnet as snt
import tensorflow as tf
from collections import namedtuple

DNCState = namedtuple('DNCState', ('memory_output', 'memory_state',
                                   'controller_state'))

class DNC(snt.RNNCore):
    """DNC core module."""
    
    def __init__(self,
                 controller_config,
                 memory_config,
                 output_size,
                 name='dnc'):
        """Initializes the DNC core.
        
        Args:
            controller_config: dictionary containing controller module
                configurations.
            memory_config: dictionary containing memory module configurations.
            output_size: output dimension size of this core module.
            name: module name (default 'dnc').
        """
        super(DNC, self).__init__(name=name)
        
        self._controller_config = controller_config
        self._memory_config = memory_config
        self._output_size = output_size
        
        self._memory_output_size = 0
        self._memory_state_size = 0
        self._controller_state_size = 0
        
        self._state_size = DNCState(
            memory_output=self._memory_output_size,
            memory_state=self._memory_state_size,
            controller_state=self._controller_state_size)
        
    def _build(self, inputs, prev_state):
        """Computes one timestep of computation with a TF graph with DNC core.
        
        Args:
            inputs: Tensor input.
            prev_state: A 'DNCState' containing the fields 'memory output',
                'memory_state', and 'controller_state'.
        """
        return [], prev_state
    
    @property
    def state_size(self):
        """Returns a description of the state size."""
        return self._state_size
    
    @property
    def output_size(self):
        """Returns a description of the output size."""
        return tf.TensorShape([self._output_size])
    
    def initial_state(self, batch_size, dtype=tf.float32):
        """Returns an initial state, for a batch size and data type.
        
        Args:
            batch_size: the batch size for each state.
            dtype: the data type of the elements in the state (default is
                tf.float32).
        """
        return DNCState(
            memory_output=self._memory_output_size,
            memory_state=self._memory_state_size,
            controller_state=self._controller_state_size)
