"""A Temporal Linkage implementation used in a DNC.

This Temporal Linkage is implemented as defined in the DNC architecture
in DeepMind's Nature paper:
http://www.nature.com/nature/journal/vaop/ncurrent/full/nature20101.html

Author: Austin Derrow-Pinion
"""

import collections
import sonnet as snt
import tensorflow as tf

TemporalLinkageState = collections.namedtuple('TemporalLinkageState', (
    'linkage'))

class TemporalLinkage(snt.RNNCore):
    """A Temporal Linkage matrix that keeps track of the order of writes.
    
    """
    
    def __init__(self,
                 memory_size=16,
                 name='temporal_linkage'):
        """Initializes a Temporal Linkage matrix used in a DNC.
        
        Args:
            memory_size: The number of memory slots (N in the DNC paper).
                Default value is 16.
            name: The name of the module (default 'temporal_linkage').
        """
        super(TemporalLinkage, self).__init__(name=name)
        
        self._memory_size = memory_size
        
        self._state_size = TemporalLinkageState(
            linkage=tf.TensorShape([self._memory_size, self._memory_size]))
        
    
    def _build(self, inputs, prev_state):
        """Computes one timestep of computation for the Temporal Linkage.
        
        Args:
            inputs: A Tensor.
            prev_state: An instance
        """
        return prev_state
    
    @property
    def state_size(self):
        """Returns a description of the state size."""
        return self._state_size
