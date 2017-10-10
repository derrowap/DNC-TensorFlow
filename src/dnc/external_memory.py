"""External Memory used in a DNC for read and writes.

This External Memory is implemented as defined in the DNC architecture
in DeepMind's Nature paper:
http://www.nature.com/nature/journal/vaop/ncurrent/full/nature20101.html

Author: Austin Derrow-Pinion
"""

import collections
import sonnet as snt
import tensorflow as tf

# Ensure values are greater than epsilon to avoid numerical instability.
_EPSILON = 1e-6

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
    
    def _build(self,
               read_keys,
               read_strengths,
               prev_state):
        """Computes one timestep of computation for the External Memory.
        
        Content based addressing starts with calculating the cosine
        similarity between the input read keys and the current state of
        memory. The content vector, `c_t`, at time `t` is outputed by
        taking the softmax of the product between the cosine similarity and
        the input read strengths.
        
        Args:
            read_keys: A Tensor of shape `[word_size]` containing the read
                keys from the Read Head originally emitted by the DNC
                controller. These are compared to each slot in the external
                memory for content based addressing.
            read_strengths: A Tensor of shape `[memory_size]` containing the
                strength at which the `c_t` vector should favor each slot in
                external memory. If `read_strengths[i]` is high, then
                `c_t[i]` will be greater than if `read_strengths[i]` was a
                small value. The values are bound to the interval [1, inf).
            prev_state: An instance of `ExternalMemoryState` containing the
                previous state of this External Memory.
        Returns:
            A tuple `(output, next_state)`. Where `output` is a Tensor of
            the content based addressing weightings. The `next_state` is
            an instance of `ExternalMemoryState` representing the next state
            of this External Memory after computation finishes.
        """
        repeated_read_keys = tf.reshape(tf.tile(read_keys,
                                                [self._memory_size]),
                                        [self._memory_size, self._word_size])
        content_similarity = self.cosine_similarity(repeated_read_keys,
                                                    prev_state.memory)
        c_t = tf.nn.softmax(tf.multiply(content_similarity, read_strengths))
        
        return (c_t, prev_state)
    
    def cosine_similarity(self, u, v):
        """Computes the cosine similarity between two vectors, `u` and `v`.
        
        Cosine similarity here is defined as in the DNC paper:
            D(u, v) = (u * v) / (|u| * |v|)
        The resulting similarity ranges from -1 meaning exactly opposite, to
        1 meaning exactly the same.
        
        Args:
            u: A 2-D Tensor of shape `[batch_size, word_size]`.
            v: A 2-D Tensor of shape `[batch_size, word_size]`.
        Returns:
            A Tensor of shape `[batch_size]` containing the cosine similarity
            for the two input vectors, `u` and `v`.
        """
        dot = tf.reduce_sum(tf.multiply(u, v), 1)
        norm = tf.norm(u, ord='euclidean', axis=1) * tf.norm(v, ord='euclidean', axis=1)
        return dot / (norm + tf.constant(_EPSILON))
    
    @property
    def state_size(self):
        """Returns a description of the state size."""
        return self._state_size
    
    @property
    def output_size(self):
        """Returns the output shape."""
        return tf.TensorShape([self._memory_size])
