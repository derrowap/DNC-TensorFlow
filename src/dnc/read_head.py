"""A 'Read Head' for the DNC.

This is an implementation of a 'Read Head' used by the DNC
architecture as described in DeepMind's Nature paper:
http://www.nature.com/nature/journal/vaop/ncurrent/full/nature20101.html

Author: Austin Derrow-Pinion
"""

import collections
import sonnet as snt
import tensorflow as tf

ReadHeadState = collections.namedtuple('ReadHeadState', (
    'memory', 'read_weights'))

class ReadHead(snt.RNNCore):
    """Reads from external memory using 'Temporal Linkage' and content.
    
    There are three different read modes:
        B:  Backward reads for reading the location in memory that was
            written to before the previously read location.
        C:  Content based addressing is useful for storing references to
            data.
        F:  Forward reads for reading the location in memory that was
            written to after the previously read location.
    
    The Temporal Linkage matrix, L_t in the paper, is used for both
    forward and backward reads:
        - Backwards read:
            b_t = Transpose(L_t) * w_(t-1)
        - Forwards read: 
            f_t = L_t * w_(t-1)
    
    Content based addressing uses a similarity calculation. In the paper,
    they used cosine similarity:
        D(u, v) = (u * v) / (|u| * |v|)
    A read strength, beta, and a read key, k, is emitted from the
    controller and used in the equation for content based lookup on
    memory M with N slots of memory each W words long:
        c_t = C(M, k, beta)[i] =
            exp{D(k, M[i,:]) * beta} / [sum_j exp{D(k, M[j,:]) * beta}
    
    The vectors b_t, c_t, and f_t, are computed through the different
    read mode operations. These together are used to calculate the final
    weighting through a sum with the read mode vector pi_t. If pi_t[1]
    dominates, then the read head will prioritize the backwards read mode.
    If pi_t[2] dominates, then the content based addressing will be used.
    If pi_t[3] dominates, then the forward read mode will be used.
        w_t = pi_t[1] * b_t + pi_t[2] * c_t + pi_t[3] * f_t
    
    The read head calculates the weights through these different read
    modes, but the emitted read vector, r_t, is a weighted sum of the
    contents in memory:
        r_t = Transpose(M_t) * w_t
    """
    
    def __init__(self,
                 memory_size,
                 word_size,
                 read_mode=[0., 1., 0.],
                 name='write_head'):
        """Initializes a Read Head used in a DNC.
        
        Args:
            memory_size: The number of memory slots (N in the DNC paper).
            word_size: The width of each memory slot (W in the DNC paper).
            read_mode: A softmax vector of length 3 such that the 3 values
                sum to 1. Each index represents the strength of using the
                respective read mode. Index 0 is for backwards, index 1 is
                for content based addressing, and index 2 is for backwards
                (pi_t in the DNC paper).
        """
        super(ReadHead, self).__init__(name=name)

        self.memory_size = memory_size
        self.word_size = word_size
        self.read_mode = read_mode

        self._state_size = ReadHeadState(
            memory=tf.TensorShape([self.memory_size, self.word_size]),
            read_weights=tf.TensorShape([self.memory_size]))
    
    def _build(self, inputs, prev_state):
        """Computes one timestep of computation for the Read Head.
        
        Args:
            inputs: A Tensor of shape [batch_size, input_size] emitted from
                the DNC controller. This holds data that controls what this
                read head does.
            prev_state: An instance of 'ReadHeadState' containing the
                previous state of this read head.
        Returns:
            A tuple (output, next_state). Output is a Tensor of shape
            [batch_size, word_size] representing the read vector, r_t.
            An instance of 'ReadHeadState' represents the next_state of
            this write head after computation finishes.
        """
        return prev_state
    
    @property
    def state_size(self):
        """Returns a description of the state size."""
        return self._state_size
    
    @property
    def output_size(self):
        """Returns the output shape."""
        return tf.TensorShape([self.word_size])
