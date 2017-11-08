"""A Usage vector implementation used in a DNC.

This Usage vector is implemented as defined in the DNC architecture in
DeepMind's Nature paper:
http://www.nature.com/nature/journal/vaop/ncurrent/full/nature20101.html

Author: Austin Derrow-Pinion
"""

import collections
import sonnet as snt
import tensorflow as tf

# Ensure values are greater than epsilon to avoid numerical instability.
_EPSILON = 1e-6

UsageState = collections.namedtuple('UsageState', ('usage_vector'))


class Usage(snt.RNNCore):
    """A Usage vector exposes to what extent each memory location is used.

    Every time external memory is written at some location, the usage here
    increases to a maximum of 1. Free gates are used to decrease the usage
    values to a minimum of 0.
    """

    def __init__(self,
                 memory_size=16,
                 name='usage'):
        """Initialize a Usage vector used in a DNC.

        Args:
            memory_size: The number of memory slots in the external memory.
                Written as `N` in the DNC paper. Default value is 16.
            name: The name of the module (default 'usage').
        """
        super(Usage, self).__init__(name=name)
        self._memory_size = memory_size
        self._state_size = UsageState(
            usage_vector=tf.TensorShape([self._memory_size]))

    def _build(self,
               prev_write_weightings,
               prev_read_weightings,
               free_gates,
               prev_state):
        """Compute one timestep of computation for the Usage vector.

        This updates the usage vector in the Usage state to the next timestep
        iteration.

        Args:
            prev_write_weightings: A Tensor of shape
                `[batch_size, memory_size]` containing the weights to write
                with. Represented as `w_{t-1}^w` in the DNC paper for time'
                `t-1`. If `w_{t-1}^w[i]` is 0 then nothing is written to memory
                regardless of the other parameters. Therefore it can be used to
                protect the external memory from unwanted modifications.
            prev_read_weightings: A Tensor of shape
                `[batch_size, num_reads, memory_size]` containing the previous
                read weights. This is written in the DNC paper as
                `w_{t-1}^{r,i}` for time `t-1` for read head `i`.
            free_gates: A Tensor of shape `[batch_size, num_reads]` containing
                a free gate value bounded in `[0, 1]` for each read head and
                emitted from the controller. The DNC paper writes the free
                gates as `f_t^i` for time `t` and read head `i`.
            prev_state: An instance of `TemporalLinkageState` containing the
                previous state of this Temporal Linkage.

        Returns:
            A tuple `(output, next_state)`. Where `output` is a Tensor of shape
            `[batch_size, memory_size]` containing allocation weighting vector
            for each batch. The `next_state` is an instance of `UsageState`
            containing this timestep's updated usage vector.
        """
        memory_retention_vector = self.memory_retention_vector(
            prev_read_weightings, free_gates)
        updated_usage_vector = self.updated_usage_vector(
            prev_state.usage_vector, prev_write_weightings,
            memory_retention_vector)
        allocation_weighting = self.allocation_weighting(updated_usage_vector)
        return (allocation_weighting,
                UsageState(usage_vector=updated_usage_vector))

    def updated_usage_vector(self,
                             prev_usage_vector,
                             prev_write_weightings,
                             memory_retention_vector):
        """Compute the updated usage vector for this timestep.

        The usage vector is written in the DNC paper as `u_t` for time `t`. It
        can be defined as:
            u_t = (u_{t-1} + w_{t-1}^w - u_{t-1} * w_{t-1}^w) * phi_t
        where `w_{t-1^w` are the write weightings at the previous timestep and
        `phi_t` is the memory retention vector for this timestep.

        Args:
            prev_usage_vector: A Tensor of shape `[batch_size, memory_size]`
                containing the usage vector values from the previous timestep.
                Written in the DNC paper as `u_{t-1}` for time `t-1`.
            prev_write_weightings: A Tensor of shape
                `[batch_size, memory_size]` containing the weights to write
                with. Represented as `w_{t-1}^w` in the DNC paper for time'
                `t-1`. If `w_{t-1}^w[i]` is 0 then nothing is written to memory
                regardless of the other parameters. Therefore it can be used to
                protect the external memory from unwanted modifications.
            memory_retention_vector:  A Tensor of shape
                `[batch_size, memory_size]` containing the values of the memory
                retention vector for this timestep. Written in the DNC paper as
                `phi_t` for time `t`.

        Returns:
            A Tensor of shape `[batch_size, memory_size]` containing the
            updated usage vector values for this timestep.
        """
        return (prev_usage_vector +
                prev_write_weightings -
                prev_usage_vector *
                prev_write_weightings) * memory_retention_vector

    def memory_retention_vector(self, prev_read_weightings, free_gates):
        """Compute the memory retention vector for this timestep.

        The memory retention vector in the DNC paper is written as `psi_t` for
        time `t`. The values represent how much each location in external
        memory will not be freed by the free gates.

        The memory retention vector is defined as:
                psi_t = PRODUCT_{i=1}^R (1 - f_t^i * w_{t-1}^{r,i})

        Args:
            prev_read_weightings: A Tensor of shape
                `[batch_size, num_reads, memory_size]` containing the previous
                read weights. This is written in the DNC paper as
                `w_{t-1}^{r,i}` for time `t-1` for read head `i`.
            free_gates: A Tensor of shape `[batch_size, num_reads]` containing
                a free gate value bounded in `[0, 1]` for each read head and
                emitted from the controller. The DNC paper writes the free
                gates as `f_t^i` for time `t` and read head `i`.

        Returns:
            A Tensor of shape `[batch_size, memory_size]` containing the values
            of the memory retention vector for this timestep.
        """
        # Tensor of shape `[batch_size, num_reads, 1]`
        free_gates_expanded = tf.expand_dims(free_gates, 2)
        # Tensor of shape `[batch_size, num_reads, memory_size]
        free_gates_weighted = free_gates_expanded * prev_read_weightings
        return tf.reduce_prod(1 - free_gates_weighted, axis=1)

    def allocation_weighting(self, usage_vector):
        """Compute the allocation weighting vector providing write locations.

        The allocation weighting vector is written in the DNC paper as `a_t`
        for time `t`. Since the vector is a weighting, the values sum to less
        than or equal to 1. If all usages are 1, that means each location in
        memory is very important and so the allocation weighting will be 0
        meaning no new information can be written to the external memory.

        Let `phi_t` be the sorted indices vector for the usage vector. The
        allocation weighting vector is computed as:
            a_t[phi_t[j]] = (1 - u_t[phi_t[j]])
                                * MULTIPLY_{i=1}^{j-1}(u_t[phi_t[i]])

        Args:
            usage_vector: A Tensor of shape `[batch_size, memory_size]`
                containing the usage vector values from this timestep. Written
                in the DNC paper as `u_t` for time `t`.

        Returns:
            A Tensor of shape `[batch_size, memory_size]` containing the values
            for the allocation vector for each batch of input.
        """
        # avoid NaN from tf.cumprod
        usage_vector = _EPSILON + (1 - _EPSILON) * usage_vector

        sorted_usage, indices = self.sorted_indices(usage_vector)
        non_usage = 1 - sorted_usage
        usage_cumprod = tf.cumprod(sorted_usage, axis=1, exclusive=True)
        sorted_allocation = tf.expand_dims(non_usage * usage_cumprod, 2)

        # reorder sorted_allocation back to original order in usage_vector
        flattened_allocation = tf.reshape(sorted_allocation, [-1])
        batch_size = tf.shape(usage_vector)[0]
        index_offset = tf.tile(
            tf.expand_dims(self._memory_size * tf.range(0, batch_size), 1),
            [1, self._memory_size])
        flattened_index_offset = tf.reshape(index_offset, [-1])
        flattened_indices = tf.reshape(indices, [-1]) + flattened_index_offset
        ordered_allocation = tf.gather(flattened_allocation, flattened_indices)
        return tf.reshape(ordered_allocation, [-1, self._memory_size])

    def sorted_indices(self, usage_vector):
        """Construct a list of sorted indices from the usage vector.

        The sort is in ascending order. Sorting is a non-differentiable
        function so these discontinuities must be ignored in the DNC function
        to calculate training gradients. The DNC paper seems to claim this
        is not relevant to learning and won't effect the outcome.

        The paper writes this sorted free list as `phi_t` for time `t`. Since
        it is in ascending order, `phi_t[1]` is the index of the least used
        location in the DNC external memory at time `t`.

        Args:
            usage_vector: A Tensor of shape `[batch_size, memory_size]`
                containing the usage vector values from this timestep. Written
                in the DNC paper as `u_t` for time `t`.

        Returns:
            A tuple `(values, indices)` where both are a Tensor of shape
            `[batch_size, memory_size]`. The `values` contain the usage vector
            values sorted in ascending order. The `indices` contain the indices
            for the sorted usage vector for every batch in ascending order.
        """
        values, descending_indices = tf.nn.top_k(usage_vector,
                                                 k=self._memory_size)
        return (tf.reverse(values, [-1]), tf.reverse(descending_indices, [-1]))

    @property
    def state_size(self):
        """Return a description of the state size."""
        return self._state_size
