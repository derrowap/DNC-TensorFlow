"""DNC core implementation.

This is an implementation of the complete Differentiable Neural Computer
architecture as described in DeepMind's Nature paper:
http://www.nature.com/nature/journal/vaop/ncurrent/full/nature20101.html

Author: Austin Derrow-Pinion
"""

import sonnet as snt
import tensorflow as tf

from collections import namedtuple
from .. dnc.tape_head import TapeHead

DNCState = namedtuple('DNCState', (
    'read_vectors', 'controller', 'tape_head'))


class DNC(snt.RNNCore):
    """DNC core module."""

    def __init__(self,
                 output_size,
                 memory_size=16,
                 word_size=16,
                 num_read_heads=1,
                 hidden_size=64,
                 name='dnc'):
        """Initialize the DNC core.

        Args:
            output_size: The size of the output to recevie from this network.
                Written in the DNC paper as `Y`.
            memory_size: The number of memory slots (`N` in the DNC paper).
                Default value is 16.
            word_size: The width of each memory slot (`W` in the DNC paper).
                Default value is 16.
            num_read_heads: The number of read heads is unbounded in the DNC,
                but the number of write heads was changed from unbounded in the
                Neural Turing Machine to only 1 in the DNC. Default value is 1.
            hidden_size: The hidden size of the controller LSTM in the DNC.
                Default value is 64.
            name: The name of the module (default 'dnc').
        """
        super(DNC, self).__init__(name=name)

        self._memory_size = memory_size
        self._word_size = word_size
        self._num_read_heads = num_read_heads
        self._hidden_size = hidden_size
        self._output_size = output_size

        self._interface_vector_size = \
            self._num_read_heads * self._word_size + \
            3 * self._word_size + \
            5 * self._num_read_heads + 3

        with self._enter_variable_scope():
            self._tape_head = TapeHead(memory_size=self._memory_size,
                                       word_size=self._word_size,
                                       num_read_heads=self._num_read_heads)
            self._controller = snt.LSTM(hidden_size=self._hidden_size)

        self._state_size = DNCState(
            read_vectors=self._tape_head.output_size,
            controller=self._controller.state_size,
            tape_head=self._tape_head.state_size)

    def _build(self, inputs, prev_state):
        """Compute one timestep of computation with a TF graph with DNC core.

        Args:
            inputs: Tensor input.
            prev_state: A 'DNCState' containing the fields 'memory output',
                'memory_state', and 'controller_state'.

        Returns:
            A tuple (output, next_state), where 'output' is used as the raw
            output to the task trained on. The 'next_state' is the next state
            of the DNC after computation of this timestep finishes.
        """
        # flattens tensors of any shape to shape [batch_size, -1]
        batch_flatten = snt.BatchFlatten()

        controller_input = tf.concat(
            [batch_flatten(inputs), batch_flatten(prev_state.read_vectors)], 1)

        controller_output, controller_state = self._controller(
            controller_input, prev_state.controller)

        # v_t = W_y[h_t^1; ...; h_t^L]
        controller_output_vector_network = snt.Linear(
            output_size=self._output_size)
        controller_output_vector = controller_output_vector_network(
            controller_output)

        # xi_t = W_xi[h_t^1; ...; h_t^L]
        interface_vector_network = snt.Linear(
            output_size=self._interface_vector_size)
        interface_vector = interface_vector_network(controller_output)

        read_vectors, tape_head_state = self._tape_head(
            interface_vector, prev_state.tape_head)

        # the concatenation of the current read vectors through the RW x Y
        # weight matrix W_r
        output_vector_network = snt.Linear(output_size=self._output_size)
        output_vector = output_vector_network(batch_flatten(read_vectors))
        output_vector = controller_output_vector + output_vector

        return (
            output_vector,
            DNCState(
                read_vectors=read_vectors,
                controller=controller_state,
                tape_head=tape_head_state,
            )
        )

    @property
    def state_size(self):
        """Return a description of the state size."""
        return self._state_size

    @property
    def output_size(self):
        """Return a description of the output size."""
        return tf.TensorShape([self._output_size])
