"""A Usage vector implementation used in a DNC.

This Usage vector is implemented as defined in the DNC architecture in
DeepMind's Nature paper:
http://www.nature.com/nature/journal/vaop/ncurrent/full/nature20101.html

Author: Austin Derrow-Pinion
"""

import collections
import sonnet as snt
import tensorflow as tf

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
        """
        return prev_state

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

    @property
    def state_size(self):
        """Return a description of the state size."""
        return self._state_size
