# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#==============================================================================
"""A repeat copy task."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np
import sonnet as snt
import tensorflow as tf

DatasetTensors = collections.namedtuple('DatasetTensors', ('input',
                                                           'target', 'mask'))

"""
Model for Figure:
python3 -m src.tasks.train --controller=ff --num_read_heads=1 --memory_size=10
    --word_size=3 --batch_size=10 --max_repeats=1 --gpu_usage=0.8 --num_bits=6
    --min_length=5 --max_length=5
    --checkpoint_dir=src/tasks/repeat_copy/checkpoints
    --checkpoint_basename=model-word-size-3.ckpt --checkpoint_interval=10000
    --num_training_iterations=100000
"""


def masked_sigmoid_cross_entropy(logits,
                                 target,
                                 mask,
                                 time_average=False,
                                 log_prob_in_bits=False):
    """Add ops to graph which compute the (scalar) NLL of the target sequence.

    The logits parametrize independent bernoulli distributions per time-step
    and per batch element, and irrelevant time/batch elements are masked out by
    the mask tensor.

    Args:
        logits: `Tensor` of activations for which sigmoid(`logits`) gives the
            bernoulli parameter.
        target: time-major `Tensor` of target.
        mask: time-major `Tensor` to be multiplied elementwise with cost T x B
            cost masking out irrelevant time-steps.
        time_average: optionally average over the time dimension (sum by
            default).
        log_prob_in_bits: iff True express log-probabilities in bits (default
            nats).
    Returns:
        A `Tensor` representing the log-probability of the target.
    """
    xent = tf.nn.sigmoid_cross_entropy_with_logits(labels=target,
                                                   logits=logits)
    loss_time_batch = tf.reduce_sum(xent, axis=2)
    loss_batch = tf.reduce_sum(loss_time_batch * mask, axis=0)

    batch_size = tf.cast(tf.shape(logits)[1], dtype=loss_time_batch.dtype)

    if time_average:
        mask_count = tf.reduce_sum(mask, axis=0)
        loss_batch /= (mask_count + np.finfo(np.float32).eps)

    loss = tf.reduce_sum(loss_batch) / batch_size
    if log_prob_in_bits:
        loss /= tf.log(2.)

    return loss


def bitstring_readable(data, batch_size, model_output=None, whole_batch=False):
    """Produce a human readable representation of the sequences in data.

    Args:
        data: data to be visualised
        batch_size: size of batch
        model_output: optional model output tensor to visualize alongside data.
        whole_batch: whether to visualise the whole batch. Only the first
            sample will be visualized if False

    Returns:
        A string used to visualise the data batch
    """
    def _readable(datum):
        return '+' + ' '.join(['-' if x == 0 else '%d' % x for x in datum]) + \
            '+'

    obs_batch = data.input
    targ_batch = data.target

    iterate_over = range(batch_size) if whole_batch else range(1)

    batch_strings = []
    for batch_index in iterate_over:
        obs = obs_batch[:, batch_index, :]
        targ = targ_batch[:, batch_index, :]

        obs_channels = range(obs.shape[1])
        targ_channels = range(targ.shape[1])
        obs_channel_strings = [_readable(obs[:, i]) for i in obs_channels]
        targ_channel_strings = [_readable(targ[:, i]) for i in targ_channels]

        readable_obs = 'Observations:\n' + '\n'.join(obs_channel_strings)
        readable_targ = 'Targets:\n' + '\n'.join(targ_channel_strings)
        strings = [readable_obs, readable_targ]

        if model_output is not None:
            output = model_output[:, batch_index, :]
            output_strings = [_readable(output[:, i]) for i in targ_channels]
            strings.append('Model Output:\n' + '\n'.join(output_strings))

        batch_strings.append('\n\n'.join(strings))

    return '\n' + '\n\n\n\n'.join(batch_strings)


def readable_model(model_state, batch_size, whole_batch=False):
    """Produce a readable representation of the model state.

    Args:
        model_state: A tuple given by the DNC representing the internal state
            of the model.
        batch_size: size of batch.
        whole_batch: Whether to visualize the whole batch. Only the first
            sample will be visualized if False.

    Returns:
        A string used to visualize the model state.
    """
    format_string = \
        "\nRead Weights  = {}\n" \
        "Write Weights = {}\n" \
        "Free Gate     = {}\n" \
        "Alloc. Gate   = {}\n"
    if whole_batch:
        return format_string.format(
            model_state.tape_head.read_weights,
            model_state.tape_head.write_weights,
            model_state.tape_head.free_gate,
            model_state.tape_head.alloc_gate)
    return format_string.format(
        model_state.tape_head.read_weights[-1],
        model_state.tape_head.write_weights[-1],
        model_state.tape_head.free_gate[-1],
        model_state.tape_head.alloc_gate[-1])


def arr_to_str(a):
    """Create more readable array to string."""
    output = "["
    for i in range(len(a)):
        output += str(a[i])
        if i == len(a) - 1:
            break
        output += ", "
    output += "]"
    return output


def figure_data(data, output, model_state, batch_size):
    """Create string with data from the model."""
    input_str = "\n# shape = {}\nobs_str = np.array([\n".format(
        str(data.input.shape))
    for batch_index in range(batch_size):
        obs = data.input[:, batch_index, :]
        for input_row_index in range(obs.shape[0]):
            input_str += "\t" + arr_to_str(obs[input_row_index, :]) + ",\n"
    input_str += "])"

    output_str = "# shape = {}\noutput_str = np.array([\n".format(
        str(output.shape))
    for batch_index in range(batch_size):
        out = output[:, batch_index, :]
        for out_row_index in range(out.shape[0]):
            out_row = out[out_row_index, :]
            output_str += "\t" + arr_to_str(out_row) + ",\n"
    output_str += "])"

    rw_str = "# shape = {}\nrw_str = np.array([\n".format(
        str(model_state['rw'].shape))
    for rw_t in model_state['rw']:
        rw_str += "\t" + arr_to_str(rw_t[0]) + ",\n"
    rw_str += "])"

    ww_str = "# shape = {}\nww_str = np.array([\n".format(
        str(model_state['ww'].shape))
    for ww_t in model_state['ww']:
        ww_str += "\t" + arr_to_str(ww_t) + ",\n"
    ww_str += "])"

    fg_str = "# shape = {}\nfg_str = np.array([\n\t".format(
        str(model_state['fg'].shape))
    count = 0
    for fg_t in model_state['fg']:
        fg_str += str(fg_t[0]) + ", "
        count += 1
        if count == 6:
            count = 0
            fg_str += "\n\t"
    fg_str += "])"

    ag_str = "# shape = {}\nag_str = np.array([\n\t".format(
        str(model_state['ag'].shape))
    count = 0
    for ag_t in model_state['ag']:
        ag_str += str(ag_t[0]) + ", "
        count += 1
        if count == 6:
            count = 0
            ag_str += "\n\t"
    ag_str += "])"
    return input_str + "\n" + output_str + "\n" + rw_str + "\n" + ww_str \
        + "\n" + fg_str + "\n" + ag_str + "\n"


class RepeatCopy(snt.AbstractModule):
    """Sequence data generator for the task of repeating random binary pattern.

    When called, an instance of this class will return a tuple of tensorflow
    ops (obs, targ, mask), representing an input sequence, target sequence, and
    binary mask. Each of these ops produces tensors whose first two dimensions
    represent sequence position and batch index respectively. The value in
    mask[t, b] is equal to 1 iff a prediction about targ[t, b, :] should be
    penalized and 0 otherwise.
    For each realisation from this generator, the observation sequence is
    comprised of I.I.D. uniform-random binary vectors (and some flags).
    The target sequence is comprised of this binary pattern repeated
    some number of times (and some flags). Before explaining in more detail,
    let's examine the setup pictorially for a single batch element:
    ```none
    Note: blank space represents 0.
    time ------------------------------------------>
                +-------------------------------+
    mask:       |0000000001111111111111111111111|
                +-------------------------------+
                +-------------------------------+
    target:     |                              1| 'end-marker' channel.
                |         101100110110011011001 |
                |         010101001010100101010 |
                +-------------------------------+
                +-------------------------------+
    observation:| 1011001                       |
                | 0101010                       |
                |1                              | 'start-marker' channel
                |        3                      | 'num-repeats' channel.
                +-------------------------------+
    ```
    The length of the random pattern and the number of times it is repeated in
    the target are both discrete random variables distributed according to
    uniform distributions whose parameters are configured at construction time.
    The obs sequence has two extra channels (components in the trailing
    dimension) which are used for flags. One channel is marked with a 1 at the
    first time step and is otherwise equal to 0. The other extra channel is
    zero until the binary pattern to be repeated ends. At this point, it
    contains an encoding of the number of times the observation pattern should
    be repeated. Rather than simply providing this integer number directly, it
    is normalised so that a neural network may have an easier time representing
    the number of repetitions internally. To allow a network to be readily
    evaluated on instances of this task with greater numbers of repetitions,
    the range with respect to which this encoding is normalised is also
    configurable by the user. As in the diagram, the target sequence is offset
    to begin directly after the observation sequence; both sequences are padded
    with zeros to accomplish this, resulting in their lengths being equal.
    Additional padding is done at the end so that all sequences in a minibatch
    represent tensors with the same shape.
    """

    def __init__(
        self,
        num_bits=6,
        batch_size=1,
        min_length=1,
        max_length=1,
        min_repeats=1,
        max_repeats=2,
        norm_max=10,
        log_prob_in_bits=False,
        time_average_cost=False,
        name='repeat_copy',
    ):
        """Create an instance of RepeatCopy task.

        Args:
            name: A name for the generator instance (for name scope purposes).
            num_bits: The dimensionality of each random binary vector.
            batch_size: Minibatch size per realization.
            min_length: Lower limit on number of random binary vectors in the
                observation pattern.
            max_length: Upper limit on number of random binary vectors in the
                observation pattern.
            min_repeats: Lower limit on number of times the obervation pattern
                is repeated in targ.
            max_repeats: Upper limit on number of times the observation pattern
                is repeated in targ.
            norm_max: Upper limit on uniform distribution w.r.t which the
                encoding of the number of repetitions presented in the
                observation sequence is normalised.
            log_prob_in_bits: By default, log probabilities are expressed in
                units of nats. If true, express log probabilities in bits.
            time_average_cost: If true, the cost at each time step will be
                divided by the `true`, sequence length, the number of
                non-masked time steps, in each sequence before any subsequent
                reduction over the time and batch dimensions.
        """
        super(RepeatCopy, self).__init__(name=name)

        self._batch_size = batch_size
        self._num_bits = num_bits
        self._min_length = min_length
        self._max_length = max_length
        self._min_repeats = min_repeats
        self._max_repeats = max_repeats
        self._norm_max = norm_max
        self._log_prob_in_bits = log_prob_in_bits
        self._time_average_cost = time_average_cost

    def _normalise(self, val):
        return val / self._norm_max

    def _unnormalise(self, val):
        return val * self._norm_max

    @property
    def time_average_cost(self):
        """The time average cost of this task."""
        return self._time_average_cost

    @property
    def log_prob_in_bits(self):
        """The log-probability in bits."""
        return self._log_prob_in_bits

    @property
    def num_bits(self):
        """The dimensionality of each random binary vector in a pattern."""
        return self._num_bits

    @property
    def output_size(self):
        """The dimensionality of the target tensor."""
        return self._num_bits + 1

    @property
    def batch_size(self):
        """The batch size of this task."""
        return self._batch_size

    def _build(self):
        """Implement build method which adds ops to graph."""
        # short-hand for private fields.
        min_length, max_length = self._min_length, self._max_length
        min_reps, max_reps = self._min_repeats, self._max_repeats
        num_bits = self.num_bits
        batch_size = self.batch_size

        # We reserve one dimension for the num-repeats and one for the
        # start-marker.
        full_obs_size = num_bits + 2
        # We reserve one target dimension for the end-marker.
        full_targ_size = num_bits + 1
        start_end_flag_idx = full_obs_size - 2
        num_repeats_channel_idx = full_obs_size - 1

        # Samples each batch index's sequence length and the number of
        # repeats.
        sub_seq_length_batch = tf.random_uniform(
            [batch_size], minval=min_length, maxval=max_length + 1,
            dtype=tf.int32)
        num_repeats_batch = tf.random_uniform(
            [batch_size], minval=min_reps, maxval=max_reps + 1,
            dtype=tf.int32)

        # Pads all the batches to have the same total sequence length.
        total_length_batch = sub_seq_length_batch * \
            (num_repeats_batch + 1) + 3
        max_length_batch = tf.reduce_max(total_length_batch)
        residual_length_batch = max_length_batch - total_length_batch

        obs_batch_shape = [max_length_batch, batch_size, full_obs_size]
        targ_batch_shape = [max_length_batch, batch_size, full_targ_size]
        mask_batch_trans_shape = [batch_size, max_length_batch]

        obs_tensors = []
        targ_tensors = []
        mask_tensors = []

        # Generates patterns for each batch element independently.
        for batch_index in range(batch_size):
            sub_seq_len = sub_seq_length_batch[batch_index]
            num_reps = num_repeats_batch[batch_index]

            # The observation pattern is a sequence of random binary
            # vectors.
            obs_pattern_shape = [sub_seq_len, num_bits]
            obs_pattern = tf.cast(
                tf.random_uniform(
                    obs_pattern_shape, minval=0, maxval=2, dtype=tf.int32),
                tf.float32)

            # The target pattern is the observation pattern repeated n
            # times. Some reshaping is required to accomplish the tiling.
            targ_pattern_shape = [sub_seq_len * num_reps, num_bits]
            flat_obs_pattern = tf.reshape(obs_pattern, [-1])
            flat_targ_pattern = tf.tile(flat_obs_pattern,
                                        tf.stack([num_reps]))
            targ_pattern = tf.reshape(flat_targ_pattern,
                                      targ_pattern_shape)

            # Expand the obs_pattern to have two extra channels for flags.
            # Concatenate start flag and num_reps flag to the sequence.
            obs_flag_channel_pad = tf.zeros([sub_seq_len, 2])
            obs_start_flag = tf.one_hot(
                [start_end_flag_idx], full_obs_size, on_value=1.,
                off_value=0.)
            num_reps_flag = tf.one_hot(
                [num_repeats_channel_idx],
                full_obs_size,
                on_value=self._normalise(tf.cast(num_reps, tf.float32)),
                off_value=0.)

            # note the concatenation dimensions.
            obs = tf.concat([obs_pattern, obs_flag_channel_pad], 1)
            obs = tf.concat([obs_start_flag, obs], 0)
            obs = tf.concat([obs, num_reps_flag], 0)

            # Now do the same for the targ_pattern (it only has one extra
            # channel).
            targ_flag_channel_pad = tf.zeros([sub_seq_len * num_reps, 1])
            targ_end_flag = tf.one_hot(
                [start_end_flag_idx], full_targ_size, on_value=1.,
                off_value=0.)
            targ = tf.concat([targ_pattern, targ_flag_channel_pad], 1)
            targ = tf.concat([targ, targ_end_flag], 0)

            # Concatenate zeros at end of obs and begining of targ. This
            # aligns them s.t. the target begins as soon as the obs ends.
            obs_end_pad = tf.zeros([sub_seq_len * num_reps + 1,
                                   full_obs_size])
            targ_start_pad = tf.zeros([sub_seq_len + 2, full_targ_size])

            # The mask is zero during the obs and one during the targ.
            mask_off = tf.zeros([sub_seq_len + 2])
            mask_on = tf.ones([sub_seq_len * num_reps + 1])

            obs = tf.concat([obs, obs_end_pad], 0)
            targ = tf.concat([targ_start_pad, targ], 0)
            mask = tf.concat([mask_off, mask_on], 0)

            obs_tensors.append(obs)
            targ_tensors.append(targ)
            mask_tensors.append(mask)

        # End the loop over batch index.
        # Compute how much zero padding is needed to make tensors sequences
        # the same length for all batch elements.
        residual_obs_pad = [
            tf.zeros([residual_length_batch[i], full_obs_size])
            for i in range(batch_size)
        ]
        residual_targ_pad = [
            tf.zeros([residual_length_batch[i], full_targ_size])
            for i in range(batch_size)
        ]
        residual_mask_pad = [
            tf.zeros([residual_length_batch[i]]) for i in range(batch_size)
        ]

        # Concatenate the pad to each batch element.
        obs_tensors = [
            tf.concat([o, p], 0) for o, p in zip(obs_tensors,
                                                 residual_obs_pad)
        ]
        targ_tensors = [
            tf.concat([t, p], 0) for t, p in zip(targ_tensors,
                                                 residual_targ_pad)
        ]
        mask_tensors = [
            tf.concat([m, p], 0) for m, p in zip(mask_tensors,
                                                 residual_mask_pad)
        ]

        # Concatenate each batch element into a single tensor.
        obs = tf.reshape(tf.concat(obs_tensors, 1), obs_batch_shape)
        targ = tf.reshape(tf.concat(targ_tensors, 1), targ_batch_shape)
        mask = tf.transpose(
            tf.reshape(tf.concat(mask_tensors, 0), mask_batch_trans_shape))
        return DatasetTensors(obs, targ, mask)

    def cost(self, output, task_state):
        """Calculate sigmoid cross-entropy cost."""
        targ = task_state.target
        mask = task_state.mask
        return masked_sigmoid_cross_entropy(
            output,
            targ,
            mask,
            time_average=self.time_average_cost,
            log_prob_in_bits=self.log_prob_in_bits)

    def to_string(self, output, task_state, model_state, verbose=False):
        """Return human readable version of data."""
        obs = task_state.input
        unnormalised_num_reps_flag = self._unnormalise(
            obs[:, :, -1:]).round()
        obs = np.concatenate([obs[:, :, :-1], unnormalised_num_reps_flag],
                             axis=2)
        data = task_state._replace(input=obs)
        output_str = bitstring_readable(
            data, self.batch_size, output, whole_batch=verbose)
        if verbose:
            output_str += figure_data(
                data, output, model_state, self.batch_size)
        return output_str

    def process_output(self, output, task_state, model_state):
        """Change output logits to output probabilities."""
        return tf.round(
            tf.expand_dims(task_state.mask, -1) * tf.sigmoid(output))
