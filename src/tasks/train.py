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
    methods `_build(self)`, `cost(output, task_state)`,
    `to_string(output, task_state)`, and `process_output(output, task_state)`.

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

    *   The `process_output(output, task_state)` method returns the output back
        if no processing is needed. This method processes the output passed to
        `to_string(output, task_state)`, but not to `cost(output, task_state)`.
        If the output needs to be processed in `cost(output, task_output)`,
        then that method needs to call it itself. This provides ability to
        transform the data before `to_string(output, task_state)` converts it
        to a human readable representation. For example, if the model outputs
        logits, but you need probabilitites (repeat copy task), then do that
        here.

*   The task's class has public property `output_size`. This property must be
    an integer representing the size of the output expected from the DNC model
    for each iteration of this task.
"""

import tensorflow as tf

from .. dnc.dnc import DNC
from . repeat_copy.repeat_copy import RepeatCopy

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
tf.flags.DEFINE_integer("num_training_iterations", 1000,
                        "Number of iterations to train for.")
tf.flags.DEFINE_integer("report_interval", 100,
                        "Iterations between reports (samples, valid loss).")
tf.flags.DEFINE_string("checkpoint_dir", "~/tmp/dnc", "Checkpoint directory.")
tf.flags.DEFINE_integer("checkpoint_interval", -1,
                        "Checkpointing step interval (-1 means never).")
tf.flags.DEFINE_float("gpu_usage", 0.2,
                      "The percent of gpu memory to use for each process.")

# Optimizer parameters
tf.flags.DEFINE_float("max_grad_norm", 50, "Gradient clipping norm limit.")
tf.flags.DEFINE_float("learning_rate", 1e-4, "Optimizer learning rate.")
tf.flags.DEFINE_float("optimizer_epsilon", 1e-10,
                      "Epsilon used for RMSProp optimizer.")


def get_task(task_name):
    """Instantiate a task with all valid flags that provides training data."""
    valid_tasks = ["repeat_copy"]
    instantiate_task = [
        lambda: RepeatCopy(
            num_bits=FLAGS.num_bits,
            batch_size=FLAGS.batch_size,
            min_length=FLAGS.min_length,
            max_length=FLAGS.max_length,
            min_repeats=FLAGS.min_repeats,
            max_repeats=FLAGS.max_repeats),
    ]
    return instantiate_task[valid_tasks.index(task_name)]()


def run_model(input, output_size):
    """Run the model on the given input and returns size output_size."""
    dnc_cell = DNC(output_size,
                   memory_size=FLAGS.memory_size,
                   word_size=FLAGS.word_size,
                   num_read_heads=FLAGS.num_read_heads,
                   hidden_size=FLAGS.hidden_size)

    initial_state = dnc_cell.initial_state(FLAGS.batch_size, dtype=input.dtype)

    output, _ = tf.nn.dynamic_rnn(
        cell=dnc_cell,
        inputs=input,
        time_major=True,
        initial_state=initial_state)

    return output


def get_config():
    """Return configuration for a tf.Session using a fraction of GPU memory."""
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_usage


def train():
    """Train the DNC and periodically report the loss."""
    task = get_task(FLAGS.task)

    task_state = task()

    output = run_model(task_state.observations, task.output_size)
    output_processed = task.process_output(output, task_state)
    # responsibility of task.cost to process output if desired
    train_loss = task.cost(output, task_state)

    trainable_variables = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(
        tf.gradients(train_loss, trainable_variables), FLAGS.max_grad_norm)

    global_step = tf.Variable(0, trainable=False, name='global_step')

    optimizer = tf.train.RMSPropOptimizer(
        FLAGS.learning_rate, epsilon=FLAGS.optimizer_epsilon)
    train_step = optimizer.apply_gradients(
        zip(grads, trainable_variables), global_step=global_step)

    saver = tf.train.Saver()

    if FLAGS.checkpoint_interval > 0:
        hooks = [
            tf.train.CheckpointSaverHook(
                checkpoint_dir=FLAGS.checkpoint_dir,
                save_steps=FLAGS.checkpoint_interval,
                saver=saver)
        ]
    else:
        hooks = []

    # Training time
    with tf.train.SingularMonitoredSession(
        hooks=hooks, config=get_config(), checkpoint_dir=FLAGS.checkpoint_dir,
    ) as sess:

        start_iteration = sess.run(global_step)
        total_loss = 0

        for train_iteration in range(start_iteration,
                                     FLAGS.num_training_iterations):
            _, loss = sess.run([train_step, train_loss])
            total_loss += loss

            # report periodically
            if (train_iteration + 1) % FLAGS.report_interval == 0:
                task_state_eval, output_eval = sess.run(
                    [task_state, output_processed])
                report_string = task.to_string(output_eval, task_state_eval)
                tf.logging.info(
                    "Train Iteration %d: Avg training loss: %f.\n%s",
                    train_iteration, total_loss / FLAGS.report_interval,
                    report_string)
                # reset total_loss to report the interval's loss only
                total_loss = 0

    return task


def main(unused):
    """Main method for this app."""
    tf.logging.set_verbosity(3)  # Print INFO log messages.
    train()

if __name__ == "__main__":
    tf.app.run()
