"""A script to train the DNC on implemented tasks.

You can start training the DNC model on any implemented task by executing:
> python -m src.tasks.train --task=<task_name>

TO SUPPORT NEW TASKS:
1) Import necessary code for task (follow necessary requirements listed below).
2) Create new section in flags and define any valid flags for the task.
3) In the "get_task" method, append a command-line name for the task to the end
   of the list "valid_tasks".
4) Append a lambda function to the end of the "instantiate_task" list that
   returns an instatiated object of the task using all FLAGS defined in step 2.

REQUIREMENTS FOR ALL TASKS:
*   The task's class must be a sub-class of snt.AbstractModule implementing
    methods `_build(self)`, `cost(output, task_state)`, and
    `to_string(output, task_state)`.

    *   The `_build(self)` method must return a collections.namedtuple,
        `task_state`, containing at least fields 'input'. Other fields are
        allowed to be used internally in the other methods. For example, the
        'target' field would likely be needed for supervised learning tasks to
        calculate the cost.

    *   The `cost(output, task_state)` method must return the losses for the
        model to be used in `tf.gradients(losses, trainable_variables)`.

    *   The `to_string(output, task_state)` method must return a string. This
        string will be logged to the console every time a report comes up
        during training time. Preferrably, this string provides an example
        input/output to show what the DNC model is doing.

*   The task's class has public property `output_size`. This property must be
    an integer representing the size of the output expected from the DNC model
    for each iteration of this task.
"""

import tensorflow as tf

from repeat_copy import RepeatCopy

FLAGS = tf.flags.FLAGS

# DNC parameters
tf.flags.DEFINE_integer("memory_size", 16, "The number of memory slots.")
tf.flags.DEFINE_integer("word_size", 16, "The width of each memory slot.")
tf.flags.DEFINE_integer("num_read_heads", 1,
                        "The number of memory read heads.")
tf.flags.DEFINE_integer("hidden_size", 64,
                        "The size of LSTM hidden layer in the controller.")

# Task parameters
tf.flags.DEFINE_integer("batch_size", 16, "The batch size used in training.")
tf.flags.DEFINE_string("task", "repeat_copy", "The task to train the DNC on.")

# RepeatCopy task parameters (used only if using the RepeatCopy task)
tf.flags.DEFINE_integer("num_bits", 4,
                        "Dimensionality of each vector to copy.")
tf.flags.DEFINE_integer("min_length", 1,
                        "Lower limit on number of vectors in the observation "
                        "pattern to copy.")
tf.flags.DEFINE_integer("max_length", 2,
                        "Upper limit on number of vectors in the observation "
                        "pattern to copy.")
tf.flags.DEFINE_integer("min_repeats", 1,
                        "Lower limit on number of copy repeats.")
tf.flags.DEFINE_integer("max_repeats", 2,
                        "Upper limit on number of copy repeats.")

# Training parameters
tf.flags.DEFINE_integer("num_training_iterations", 10000,
                        "Number of iterations to train for.")
tf.flags.DEFINE_integer("report_interval", 100,
                        "Iterations between reports (samples, valid loss).")
tf.flags.DEFINE_string("checkpoint_dir", "~/tmp/dnc", "Checkpoint directory.")
tf.flags.DEFINE_integer("checkpoint_interval", -1,
                        "Checkpointing step interval (-1 means never).")


def get_task(task_name):
    """Instantiate a task with all valid flags that provides training data."""
    valid_tasks = ["repeat_copy"]
    instantiate_task = [
        lambda: RepeatCopy(
            num_bits=FLAGS.num_bits,
            min_length=FLAGS.min_length,
            max_length=FLAGS.max_length,
            min_repeats=FLAGS.min_repeats,
            max_repeats=FLAGS.max_repeats),
    ]
    return instantiate_task[valid_tasks.indexof(task_name)]()
