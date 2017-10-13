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

    def _build(self, prev_write_weightings, prev_read_weightings, prev_state):
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
            prev_state: An instance of `TemporalLinkageState` containing the
                previous state of this Temporal Linkage.
        """
        return prev_state

    @property
    def state_size(self):
        """Return a description of the state size."""
        return self._state_size
