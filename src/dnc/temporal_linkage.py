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
    'linkage_matrix', 'precedence_weights'))


class TemporalLinkage(snt.RNNCore):
    """A Temporal Linkage matrix that keeps track of the order of writes.

    This allows the DNC to iterate forward or backward in sequences of data
    written to external memory. This is very important for the DNC to be able
    to accomplish many tasks. For example, write a sequence of instructions to
    memory to later be executed in order.
    """

    def __init__(self,
                 memory_size=16,
                 name='temporal_linkage'):
        """Initialize a Temporal Linkage matrix used in a DNC.

        Args:
            memory_size: The number of memory slots in the external memory.
                Written as `N` in the DNC paper. Default value is 16.
            name: The name of the module (default 'temporal_linkage').
        """
        super(TemporalLinkage, self).__init__(name=name)
        self._memory_size = memory_size
        self._state_size = TemporalLinkageState(
            linkage_matrix=tf.TensorShape([self._memory_size,
                                           self._memory_size]),
            precedence_weights=tf.TensorShape([self._memory_size]))

    def _build(self, write_weightings, prev_state):
        """Compute one timestep of computation for the Temporal Linkage.

        Args:
            write_weightings: A Tensor of shape `[batch_size, memory_size]`
                containing the weights to write with. Represented as `w_t^w`
                in the DNC paper for time `t`. If `w_t^w[i]` is 0 then nothing
                is written to memory regardless of the other parameters.
                Therefore it can be used to protect the external memory from
                unwanted modifications.
            prev_state: An instance of `TemporalLinkageState` containing the
                previous state of this Temporal Linkage.
        """
        new_linkage = self.updated_temporal_linkage_matrix(
            write_weightings, prev_state.precedence_weights,
            prev_state.linkage_matrix)
        new_precedence_weights = self.updated_precedence_weights(
            write_weightings, prev_state.precedence_weights)
        return TemporalLinkageState(linkage_matrix=new_linkage,
                                    precedence_weights=new_precedence_weights)

    def directional_weights(self, linkage, prev_read_weights):
        """Compute the forward and backward weighting for the DNC read heads.

        Given the temporal linkage matrix, `L_t`, and the previous timestep's
        read weightings, `w_{t-1}^{r,i}`, forward and backward weightings can
        be calculated as:
            f_t^i = L_t * w_{t-1}^{r,i}
            b_t^i = TRANSPOSE(L_t) * w_{t-1}^{r,i}

        Args:
            linkage: A Tensor of shape `[batch_size, memory_size, memory_size]`
                containing the values in the temporal linkage matrix.
                Represented in the DNC paper as `L_t` were `L_t[i, j]` is the
                degree to which slot `i` in the external memory matrix was the
                location written to after location `j`.
            prev_read_weights: A Tensor of shape
                `[batch_size, num_reads, memory_size]` containing the previous
                read weights. This is written in the DNC paper as
                `w_{t-1}^{r,i}` for time `t-1` for read head `i`.
        Returns:
            A tuple `(f_t^i, b_t^i)`. `f_t^i` is a Tensor of shape
            `[batch_size, num_reads, memory_size]` containing the values for
            forward weighting on every read head. `b_t^i` is a Tensor of shape
            `[batch_size, num_reads, memory_size]` containing the values for
            backward weighting on every read head.
        """
        f_t = tf.matmul(linkage, prev_read_weights, transpose_b=True)
        b_t = tf.matmul(linkage, prev_read_weights, transpose_a=True,
                                                    transpose_b=True)
        return (tf.matrix_transpose(f_t), tf.matrix_transpose(b_t))

    def updated_temporal_linkage_matrix(self,
                                        write_weightings,
                                        precedence_weights,
                                        linkage):
        """Compute the next timestep values for the temporal linkage matrix.

        The temporal linkage matrix, represented in the DNC paper as `L_t`
        is defined by the following recurrence relation:
                L_0[i, j] = 0, for all i, j
                L_t[i, i] = 0, for all i
                L_t[i, j] = (1 - w_t^w[i] - w_t^w[j]) * L_{t - 1}[i, j]
                                + w_t^w[i] * p_{t - 1}[j]

        Args:
            write_weightings: A Tensor of shape `[batch_size, memory_size]`
                containing the weights to write with. Represented as `w_t^w`
                in the DNC paper for time `t`. If `w_t^w[i]` is 0 then nothing
                is written to memory regardless of the other parameters.
                Therefore it can be used to protect the external memory from
                unwanted modifications.
            precedence_weights: A Tensor of shape `[batch_size, memory_size]`.
                Represented in the DNC paper as `p_t` for time `t`. The value
                `p_t[i]` represents the degree to which location `i` was the
                last slot in external memory to be written to.
            linkage: A Tensor of shape `[batch_size, memory_size, memory_size]`
                containing the values in the temporal linkage matrix.
                Represented in the DNC paper as `L_t` were `L_t[i, j]` is the
                degree to which slot `i` in the external memory matrix was the
                location written to after location `j`.
        Returns:
            A Tensor of shape `[batch_size, memory_size, memory_size]`
            containing the next timestep values for the temporal linkage
            matrix.
        """
        w_t_i = tf.expand_dims(write_weightings, 2)
        w_t_j = tf.expand_dims(write_weightings, 1)
        p_t_j = tf.expand_dims(precedence_weights, 1)
        weight_differences = 1 - w_t_i - w_t_j
        weighted_precedence = w_t_i * p_t_j
        new_linkage = weight_differences * linkage + weighted_precedence
        batch_size = linkage.get_shape()[0].value
        updated_linkage = tf.matrix_set_diag(
            new_linkage,
            tf.zeros([batch_size, self._memory_size], dtype=linkage.dtype))
        return updated_linkage

    def updated_precedence_weights(self, write_weightings, precedence_weights):
        """Compute the next timestep value for the precedence weights.

        `p_t` is defined in the DNC paper as the following recurrence
        relationship where `w_t^w` is the write weightings at time `t`:
                p_0 = (p_{0,0}, p_{0,1}, ..., p_{0, N}) = (0, 0, ..., 0)
                p_t = (1 - SUM_i(w_t^w[i])) * p_{t-1} + w_t^w

        Args:
            write_weightings: A Tensor of shape `[batch_size, memory_size]`
                containing the weights to write with. Represented as `w_t^w`
                in the DNC paper for time `t`. If `w_t^w[i]` is 0 then nothing
                is written to memory regardless of the other parameters.
                Therefore it can be used to protect the external memory from
                unwanted modifications.
            precedence_weights: A Tensor of shape `[batch_size, memory_size]`.
                Represented in the DNC paper as `p_t` for time `t`. The value
                `p_t[i]` represents the degree to which location `i` was the
                last slot in external memory to be written to.
        Returns:
            A Tensor of shape `[batch_size, memory_size]` containing the next
            timestep values for the precedence weights as defined by the
            recurrence relation for `p_t`.
        """
        # A Tensor of shape `[batch_size, 1]`
        subtracted_write_weights = 1 - tf.reduce_sum(write_weightings,
                                                     axis=1,
                                                     keep_dims=True)
        # A Tensor of shape `[batch_size, memory_size]`
        p_t = subtracted_write_weights * precedence_weights + write_weightings
        return p_t

    @property
    def state_size(self):
        """Return a description of the state size."""
        return self._state_size
