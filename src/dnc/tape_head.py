"""A Tape Head to the DNC.

This 'Tape Head' contains a 'Write Head' and a 'Read Head' as defined in the
DNC architecutre in DeepMind's Nature paper:
http://www.nature.com/nature/journal/vaop/ncurrent/full/nature20101.html

Author: Austin Derrow-Pinion
"""

import collections
import numpy as np
import sonnet as snt
import tensorflow as tf

from .. dnc.external_memory import ExternalMemory
from .. dnc.temporal_linkage import TemporalLinkage
from .. dnc.usage import Usage

TapeHeadState = collections.namedtuple('TapeHeadState', (
    'memory', 'read_weights', 'write_weights', 'linkage', 'usage'))


class TapeHead(snt.RNNCore):
    """A Tape Head that utilizes any number of read heads and one write head.

    A normal computer simply writes to memory the exact value at an exact
    address in memory. This is differentiable and therefore trainable to
    write at useful locations in memory for a given task.

    Temporal Linkage is used to keep track of the order things are written in.
    This allows the write head to iterate forward and backwards through
    temporally related elements written in memory.

    Usage keeps track of what memory locations have and have not been used
    recently. If a memory location has not been used for a long time, the
    controller learns to free up this space to make it available for future
    writes to memory.

    There exist three different read modes:
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
                 memory_size=16,
                 word_size=16,
                 num_read_heads=1,
                 name='tape_head'):
        """Initialize a Tape Head used in a DNC.

        Args:
            memory_size: The number of memory slots (N in the DNC paper).
                Default value is 16.
            word_size: The width of each memory slot (W in the DNC paper).
                Default value is 16.
            num_read_heads: The number of read heads is unbounded in the DNC,
                but the number of write heads was changed from unbounded in the
                Neural Turing Machine to only 1 in the DNC. Default value is 1.
            name: The name of the module (default 'tape_head').
        """
        super(TapeHead, self).__init__(name=name)

        self._memory_size = memory_size
        self._word_size = word_size
        self._num_read_heads = num_read_heads

        # TODO(derrowap): when linkage and usage are implemented, use their
        # state size properties here instead of 'None'.
        self._state_size = TapeHeadState(
            memory=tf.TensorShape([self._memory_size, self._word_size]),
            read_weights=tf.TensorShape([self._memory_size]),
            write_weights=tf.TensorShape([self._memory_size]),
            linkage=None,
            usage=None)

    def _build(self, inputs, prev_state):
        """Compute one timestep of computation for the Tape Head.

        Args:
            inputs: A Tensor of shape `[batch_size, num_read_heads *
                word_size + 3 * word_size + 5 * num_read_heads + 3]` emitted
                from the DNC controller. This holds data that controls what
                this read head does.
            prev_state: An instance of 'TapeHeadState' containing the
                previous state of this Tape Head.

        Returns:
            A tuple `(output, next_state)`. Where `output` is a Tensor of
            shape `[batch_size, word_size]` representing the read vector
            result, `r_t`. The `next_state` is an instance of `TapeHeadState`
            representing the next state of this Tape Head after computation
            finishes.
        """
        (
            read_keys, read_strengths, write_key, write_strengths,
            erase_vector, write_vector, free_gates, allocation_gate,
            write_gate, read_modes
        ) = self.interface_parameters(inputs)
        external_memory = ExternalMemory(memory_size=self._memory_size,
                                         word_size=self._word_size)
        usage = Usage(memory_size=self._memory_size)
        temporal_linkage = TemporalLinkage(memory_size=self._memory_size)

        allocation_weighting = usage(prev_state.write_weights,
                                     prev_state.read_weights,
                                     free_gates,
                                     prev_state.usage)
        write_content_weighting = external_memory.content_weights(
            read_keys, read_strengths, prev_state.memory.memory)
        write_weighting = self.write_weighting(write_gate,
                                               allocation_gate,
                                               allocation_weighting,
                                               write_content_weighting)
        content_weighting, memory_next_state = external_memory(
            read_keys,
            read_strengths,
            write_weighting,
            erase_vector,
            write_vector,
            prev_state.memory)

        linkage = temporal_linkage(write_weighting, prev_state.linkage)

        (forward_weighting,
         backward_weighting) = temporal_linkage.directional_weights(
            linkage.linkage_matrix, prev_state.read_weights)

        read_weights = self.read_weights(read_modes, backward_weighting,
                                         content_weighting, forward_weighting)
        read_vectors = tf.matmul(memory_next_state.memory, read_weights,
                                 transpose_a=True)
        return (read_vectors, prev_state)

    def write_weighting(self,
                        write_gate,
                        allocation_gate,
                        allocation_weighting,
                        write_content_weighting):
        """Compute the write weighting vector.

        The write weighting vector, `w_t^w` is defined as:
                w_t^w = g_t^w * [g_t^a * a_t + (1 - g_t^a) * c_t^w]

        Where `g_t^w` is the write gate, `g_t^a` is the allocation gate, `a_t`
        is the allocation weighting, and `c_t^w` is the write content
        weighting.

        Args:
            write_gate: A Tensor of shape `[batch_size, 1]` containing the
                write gate values from the interface parameters. Written as
                `g_t^w` in the DNC paper.
            allocation_gate: A Tensor of shape `[batch_size, 1]` containing the
                allocation gate values from the interface parameters. Written
                as `g_t^a` in the DNC paper.
            allocation_weighting: A Tensor of shape `[batch_size, memory_size]`
                containing the allocation weighting values from the Usage
                class. Written as `a_t` in the DNC paper.
            write_content_weighting: A Tensor of shape
                `[batch_size, memory_size]` containing the write content
                weighting values. Written as `c_t^w` in the DNC paper.

        Returns:
            A Tensor of shape `[batch_size, memory_size]` containing the write
            weighting values. Written as `w_t^w` in the DNC paper.
        """
        # [batch_size, 1]
        output = 1 - allocation_gate
        # [batch_size, memory_size]
        output = output * write_content_weighting
        # [batch_size, memory_size]
        output = output + allocation_gate * allocation_weighting
        # [batch_size, memory_size]
        output = output * write_gate
        return output

    def read_weights(self,
                     read_modes,
                     backward_weighting,
                     content_weighting,
                     forward_weighting):
        """Compute the read weighting vector.

        The read weighting vector is written in the DNC paper as `w_t^{r,i}`
        for time `t` and read head `i`. This can be calculated by:
            w_t^{r,i} = pi_t^i[0] * b_t^i
                        + pi_t^i[1] * c_t^{r,i}
                        + pi_t^i[2] * f_t^{r,i}

        Args:
            read_modes: A Tensor of shape `[batch_size, num_read_heads, 3]`
                containing the read modes emitted by the DNC controller.
            backward_weighting: A Tensor of shape
                `[batch_size, num_read_heads, memory_size]` containing the
                values for the backward weighting.
            content_weighting: A Tensor of shape
                `[batch_size, num_read_heads, memory_size]` containing the
                values for the content weighting.
            forward_weighting: A Tensor of shape
                `[batch_size, num_read_heads, memory_size]` containing the
                values for the forward weighting.

        Returns:
            A Tensor of shape `[batch_size, num_read_heads, memory_size]`.
        """
        # [[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        #  [[0.7, 0.8, 0.9], [0.8, 0.7, 0.6]]]
        backward_mode = tf.slice(read_modes, [0, 0, 0], [-1, -1, 1])
        content_mode = tf.slice(read_modes, [0, 0, 1], [-1, -1, 1])
        forward_mode = tf.slice(read_modes, [0, 0, 2], [-1, -1, 1])
        #   [batch_size, num_read_heads, 1]
        # * [batch_size, num_read_heads, memory_size]
        # = [batch_size, num_read_heads, memory_size]
        return tf.multiply(backward_mode, backward_weighting) + \
            tf.multiply(content_mode, content_weighting) + \
            tf.multiply(forward_mode, forward_weighting)

    def interface_parameters(self, interface_vector):
        """Extract the interface parameters from the interface vector.

        The interface vector is written as `xi_t` in the DNC paper for time
        `t`. Below is the equation of what the interface vector contains:
            xi_t = [
                k_t^{r,1}; ...; k_t^{r,R};          (R read keys)
                beta_t^{r,1}; ...; beta_t^{r,R};    (R read strenghts)
                k_t^w;                              (the write key)
                beta_t^w;                           (the write strength)
                e_t;                                (the erase vector)
                v_t;                                (the write vector)
                f_t^1; ...; f_t^R;                  (R free gates)
                g_t^a;                              (the allocation gate)
                g_t^w;                              (the write gate)
                pi_t^1; ...; pi_t^R                 (R read modes)
            ]

        The read and write strengths are processed with the `oneplus` function
        to restrict the values in the domain `[1, infinity)`:
            oneplus(x) = 1 + log(1 + exp(x))

        The erase vector, free gates, allocation gate, and write gate, are
        processed with the logistic sigmoid function to constrain the values to
        the domain `[0, 1]`.

        The read modes are processed with the softmax function so that for any
        read mode, `pi_t^i`, the values are bound to the domain `[0, 1]` and
        the sum of the elements in the vector is equal to 1.

        Args:
            interface_vector: A Tensor of shape `[batch_size, num_read_heads *
                word_size + 3 * word_size + 5 * num_read_heads + 3]` containing
                the individual components emitted by the DNC controller. This
                is written in the DNC paper as `xi_t` for time `t`.

        Returns:
            A tuple `(read_keys, read_strengths, write_key, write_strengths,
            erase_vector, write_vector, free_gates, allocation_gate,
            write_gate, read_modes)` as explained in the description of this
            method.
        """
        def _get(shape, offset):
            size = np.prod(shape)
            output = interface_vector[:, offset:offset + size]
            return tf.reshape(output, shape=[-1] + shape), offset + size

        def _oneplus(x):
            return 1 + tf.log(1 + tf.exp(x))

        offset = 0
        read_keys, offset = _get([self._num_read_heads, self._memory_size],
                                 offset)
        _read_strengths, offset = _get([self._num_read_heads], offset)
        write_key, offset = _get([self._word_size], offset)
        _write_strengths, offset = _get([1], offset)
        _erase_vector, offset = _get([self._word_size], offset)
        write_vector, offset = _get([self._word_size], offset)
        _free_gates, offset = _get([self._num_read_heads], offset)
        _allocation_gate, offset = _get([1], offset)
        _write_gate, offset = _get([1], offset)
        _read_modes, offset = _get([self._num_read_heads, 3], offset)

        read_strengths = _oneplus(_read_strengths)
        write_strengths = _oneplus(_write_strengths)

        erase_vector = tf.sigmoid(_erase_vector)
        free_gates = tf.sigmoid(_free_gates)
        allocation_gate = tf.sigmoid(_allocation_gate)
        write_gate = tf.sigmoid(_write_gate)

        read_modes = tf.nn.softmax(_read_modes)

        return (
            read_keys, read_strengths, write_key, write_strengths,
            erase_vector, write_vector, free_gates, allocation_gate,
            write_gate, read_modes
        )

    @property
    def state_size(self):
        """Return a description of the state size."""
        return self._state_size

    @property
    def output_size(self):
        """Return the output shape."""
        return tf.TensorShape([self._word_size])
