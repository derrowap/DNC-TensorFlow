"""External Memory used in a DNC for read and writes.

This External Memory is implemented as defined in the DNC architecture
in DeepMind's Nature paper:
http://www.nature.com/nature/journal/vaop/ncurrent/full/nature20101.html

Author: Austin Derrow-Pinion
"""

import collections
import tensorflow as tf
import sonnet as snt

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
        """Initialize External Memory used in a DNC.

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
               write_weightings,
               erase_vector,
               write_vector,
               prev_state):
        """Compute one timestep of computation for the External Memory.

        Args:
            read_keys: A Tensor of shape `[batch_size, num_reads, word_size]`
                containing the read keys from the Read Head originally emitted
                by the DNC controller. These are compared to each slot in the
                external memory for content based addressing. Represented as
                `k_t^{r,i}` in the DNC paper for the read keys at time `t`
                and read head `i`.
            read_strengths: A Tensor of shape `[batch_size, num_reads]`
                containing the key strength at which the `c_t` vector should
                be heightened in value. The values are bound to the interval
                [1, inf). Represented in the DNC paper as `beta_t^{r,i}` for
                time `t` with the ith read head.
            write_weightings: A Tensor of shape `[batch_size, memory_size]`
                containing the write weightings for external memory write
                operation. If `write_weighting[,i]` is high, then that means
                the ith slot in external memory should be written to the value
                in `write_vector[i]` much more than if the value was low.
                Represented as w_t^w in the DNC paper for write weightings
                at time `t`.
            erase_vector: A Tensor of shape `[batch_size, word_size]` with
                the weights to erase from external memory. If
                `erase_vector[,i]` is high then the ith word in each slot of
                external memory will be erased more than if the value was
                smaller. Represented as `e_t` in the DNC paper for the erase
                vector at time `t`.
            write_vector: A Tensor of shape `[batch_size, word_size]` with
                the values to write in memory. This vector was directly
                emitted from the DNC controller. Represented as `v_t` in the
                DNC paper for the write vector at time `t`.
            prev_state: An instance of `ExternalMemoryState` containing the
                previous state of this External Memory.
        Returns:
            A tuple `(output, next_state)`. Where `output` is a Tensor of the
            content based addressing weightings. The `next_state` is an
            instance of `ExternalMemoryState` representing the next state of
            this External Memory after computation finishes.
        """
        # content adressing weightings
        c_t = self.content_weights(
            read_keys, read_strengths, prev_state.memory)

        # write to memory
        memory = self.write(
            write_weightings, erase_vector, write_vector, prev_state.memory)

        return (c_t, ExternalMemoryState(memory=memory))

    def content_weights(self, read_keys, read_strengths, memory):
        """Calculate content based addressing weights, c_t.

        Content based addressing starts with calculating the cosine similarity
        between the input read keys and the current state of memory. The
        content vector, `c_t`, at time `t` is outputed by taking the softmax of
        the product between the cosine similarity and the input read strengths.

        Cosine similarity here is defined as in the DNC paper:
            D(u, v) = (u * v) / (|u| * |v|)
        The resulting similarity ranges from -1 meaning exactly opposite, to
        1 meaning exactly the same.

        Args:
            read_keys: A Tensor of shape `[batch_size, num_reads, word_size]`
                containing the read keys from the Read Head originally emitted
                by the DNC controller. These are compared to each slot in the
                external memory for content based addressing.
            read_strengths: A Tensor of shape `[batch_size, num_reads]`
                containing the key strength at which the `c_t` vector should
                be heightened in value. The values are bound to the interval
                [1, inf). Represented in the DNC paper as `beta_t^{r,i}` for
                time `t` with the ith read head.
            memory: A Tensor of shape `[batch_size, memory_size, word_size]`
                containing the data of the external memory.
        Returns:
            A Tensor of shape `[batch_size, num_reads, memory_size]` with the
            content weights for each of the num_reads many read heads.
        """
        # Sneaky way of doing batched dot product between read_keys and memory.
        # Each read key, k_t^{r,i}, is element-wise multiplied against each row
        # of memory and summed together (definition of dot product). This
        # outputs a Tensor of size `[batch_size, num_reads, memory_size]`.
        dot = tf.matmul(read_keys, memory, transpose_b=True)

        # Euclidean norms
        read_keys_norm = tf.norm(
            read_keys, ord='euclidean', axis=2, keep_dims=True)
        memory_norm = tf.norm(memory, ord='euclidean', axis=2, keep_dims=True)
        norm = tf.matmul(read_keys_norm, memory_norm, transpose_b=True)

        # similarity of each read key vector from each read head in every batch
        content_similarity = dot / (norm + tf.constant(_EPSILON))

        # Content weighting is a softmax of the content similarity.
        # The product of content_similarity and read_strenghts uses
        # broadcasting to handle unequal Tensor shapes.
        return tf.nn.softmax(tf.multiply(content_similarity,
                                         tf.expand_dims(read_strengths, 2)))

    def write(self, write_weightings, erase_vector, write_vector, memory):
        """Write to external memory through en erase then add operation.

        The write operation to external memory consists of two parts: erase
        operation, add operation.

        Let `M_t(i)` be the external memory matrix contents of slot `i` at time
        `t`. Let `w_t(i)` be the write weightings vector at time `t` for the
        slot `i` in external memory. Let `e_t` be the erase vector at time `t`.
        Let `v_t` be the write vector at time `t` that is added to memory.

        Erase Operation:
            M_t'(i) = M_{t-1}(i) * (1 - w_t(i) * e_t)

        Add Operation:
            M_t(i) = M_t'(i) + w_t(i) * v_t

        Args:
            write_weightings: A Tensor of shape `[batch_size, memory_size]`
                containing the write weightings for external memory write
                operation. If `write_weighting[,i]` is high, then that means
                the ith slot in external memory should be written to the value
                in `write_vector[i]` much more than if the value was low.
                Represented as w_t^w in the DNC paper for write weightings
                at time `t`.
            erase_vector: A Tensor of shape `[batch_size, word_size]` with
                the weights to erase from external memory. If
                `erase_vector[,i]` is high then the ith word in each slot of
                external memory will be erased more than if the value was
                smaller. Represented as `e_t` in the DNC paper for the erase
                vector at time `t`.
            write_vector: A Tensor of shape `[batch_size, word_size]` with
                the values to write in memory. This vector was directly
                emitted from the DNC controller. Represented as `v_t` in the
                DNC paper for the write vector at time `t`.
            memory: A Tensor of shape `[batch_size, memory_size, word_size]`
                containing the data of the external memory.
        Returns:
            A Tensor of shape `[batch_size, memory_size, word_size]` containing
            the contents of memory after writing for each batch.
        """
        # Tensor of shape `[batch_size, memory_size, 1]`
        write_weightings_expanded = tf.expand_dims(write_weightings, 2)
        # Tensor of shape `[batch_size, 1, word_size]`
        erase_vector_expanded = tf.expand_dims(erase_vector, 1)
        # Product is Tensor of shape `[batch_size, memory_size, word_size]`
        weighted_erase_vector = tf.matmul(write_weightings_expanded,
                                          erase_vector_expanded)
        # Erase memory
        erased_memory = tf.multiply(memory, 1 - weighted_erase_vector)
        # Tensor of shape `[batch_size, 1, word_size]`
        write_vector_expanded = tf.expand_dims(write_vector, 1)
        # Product is Tensor of shape `[batch_size, memory_size, word_size]`
        weighted_write_vector = tf.matmul(write_weightings_expanded,
                                          write_vector_expanded)
        # Write to memory
        written_memory = tf.add(erased_memory, weighted_write_vector)
        return written_memory

    @property
    def state_size(self):
        """Return a description of the state size."""
        return self._state_size

    @property
    def output_size(self):
        """Return the output shape."""
        return tf.TensorShape([self._memory_size])
