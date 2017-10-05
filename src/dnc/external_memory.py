"""External Memory used in a DNC for read and writes.

This External Memory is implemented as defined in the DNC architecture
in DeepMind's Nature paper:
http://www.nature.com/nature/journal/vaop/ncurrent/full/nature20101.html

Author: Austin Derrow-Pinion
"""

import collections
import sonnet as snt
import tensorflow as tf

ExternalMemoryState = collections.namedtuple('ExternalMemoryState', (
    'memory'))

class ExternalMemory(snt.RNNCore):
    """External Memory for the DNC to read and write to.
    
    In a content based addressing read mode, a DNC read head compares
    a read vector to the contents stored in this External Memory. This
    similarity is computed using cosine similarity in the DNC paper.
    
    A write head writes to external memory in a DNC through first an
    erase operation and then an adding operation. The erase operation
    adds an erase vector across each memory slot and is then followed
    by adding the write vector to memory.
    """
    
    def __init__(self,
                 memory_size=16,
                 word_size=16,
                 name='external_memory'):
        """Initializes External Memory used in a DNC.

        Args:
            memory_size: The number of memory slots (N in the DNC paper).
                Default value is 16.
            word_size: The width of each memory slot (W in the DNC paper).
                Default value is 16.
            name: The name of the module (default 'external_memory').
        """
        super(ExternalMemory, self).__init__(name=name)
        
        self._memory_size = memory_size
        self._word_size = word_size
        
        self._state_size = ExternalMemoryState(
            memory=tf.TensorShape([self._memory_size, self._word_size]))
    
    def _build(self, inputs, prev_state):
        """Computes one timestep of computation for the External Memory.
        
        Args:
            inputs: A Tensor of shape [batch_size, input_size] emitted from
                the DNC controller. This holds data that controls what this
                external memory should do.
            prev_state: An instance of 'ExternalMemoryState' containing the
                previous state of this External Memory.
        Returns:
            A tuple `(output, next_state)`. Where `output` is a Tensor of
            the content based addressing weightings. The `next_state` is
            an instance of 'TapeHeadState' representing the next state of
            this Tape Head after computation finishes.
        """
        return prev_state
    
    @property
    def state_size(self):
        """Returns a description of the state size."""
        return self._state_size
