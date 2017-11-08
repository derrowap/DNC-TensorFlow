# DNC-TensorFlow
An implementation of the Differential Neural Computer (DNC) in [TensorFlow](https://www.tensorflow.org/), using [Sonnet](https://github.com/deepmind/sonnet), introduced in DeepMind's Nature paper:
> [“Hybrid computing using a neural network with dynamic external memory", Nature 538, 471–476 (October 2016) doi:10.1038/nature20101.](http://www.nature.com/articles/nature20101.epdf?author_access_token=ImTXBI8aWbYxYQ51Plys8NRgN0jAjWel9jnR3ZoTv0MggmpDmwljGswxVdeocYSurJ3hxupzWuRNeGvvXnoO8o4jTJcnAyhGuZzXJ1GEaD-Z7E6X_a9R-xqJ9TfJWBqz)

Training
--------
Currently only the Repeat-Copy task is implemented, but I have plans for more.

Every task has default flags available to them for changing how to train the DNC on the specified task. Every task has their own available parameters described in their own section. Below is a description of all currently available non-task-specific parameters with their default values.
```
/DNC-TensorFlow$ python3 -m src.tasks.train \
> --task=repeat_copy # The task to train the DNC on \
> --memory_size=16 # The number of memory slots \
> --word_size=16 # The width of each memory slot \
> --num_read_heads=1 # The number of memory read heads \
> --hidden_size=64 # The size of LSTM hidden layer in the controller \
> --batch_size=16 # The batch size used in training \
> --num_training_iterations=1000 # Number of iterations to train for \
> --report_interval=100 # Iterations between reports (samples, valid loss) \
> --checkpoint_dir=~/tmp/dnc # Checkpoint directory \
> --checkpoint_interval=-1 # Checkpointing step interval (-1 means never) \
> --gpu_usage=0.2 # The percent of gpu memory to use for each process \
> --max_grad_norm=50 # Gradient clipping norm limit \
> --learning_rate=1e-4 # Optimizer learning rate
> --optimizer_epsilon=1e-10 # Epsilon used for RMSProp optimizer
```

### Repeat Copy
Reference the paper for a description of this task. Essentially, the DNC learns to copy a sequence of binary digits for some number of times. It tests the DNC's ability to store useful information in the external memory, reallocate for new sequences, and not forget sequences as it needs to repeat an increasing number of times.

To execute the training script:
```
/DNC-TensorFlow$ python3 -m src.tasks.train --task=repeat_copy \
> --num_bits=4 # Dimensionality of each vector to copy \
> --min_length=1 # Lower limit on number of vectors in the observation pattern to copy \
> --max_length=2 # Upper limit on number of vectors in the observation pattern to copy \
> --min_repeats=1 # Lower limit on number of copy repeats \
> --max_repeats=2 # Upper limit on number of copy repeats
```

Testing
-------
You can run all the tests for DNC by executing:
```
/DNC-TensorFlow$ python3 -m src.run_tests
```

Alternatively, you can execute individual tests by executing:
```
/DNC-TensorFlow$ python3 -m src.testing.dnc_test
/DNC-TensorFlow$ python3 -m src.testing.external_memory_test
/DNC-TensorFlow$ python3 -m src.testing.tape_head_test
/DNC-TensorFlow$ python3 -m src.testing.temporal_linkage_test
/DNC-TensorFlow$ python3 -m src.testing.usage_test
```

Style Conformance
-----------------
All Python 3 code follows the standards enforced by the
[Python Flake8 Lint](https://github.com/dreadatour/Flake8Lint) Sublime Text 3 package. This includes:

**[Flake8](http://pypi.python.org/pypi/flake8)** (used in "Python Flake8 Lint") is a wrapper around these tools:

* **[pep8](http://pypi.python.org/pypi/pep8)** is a tool to check your Python code against some of the style conventions in [PEP8](http://www.python.org/dev/peps/pep-0008/).

* **[PyFlakes](https://launchpad.net/pyflakes)** checks only for logical errors in programs; it does not perform any check on style.

* **[mccabe](http://nedbatchelder.com/blog/200803/python_code_complexity_microtool.html)** is a code complexity checker. It is quite useful to detect over-complex code. According to McCabe, anything that goes beyond 10 is too complex. See [Cyclomatic_complexity](https://en.wikipedia.org/wiki/Cyclomatic_complexity).

There are additional tools used to lint Python files:

* **[pydocstyle](https://github.com/PyCQA/pydocstyle)** is a static analysis tool for checking compliance with Python [PEP257](http://www.python.org/dev/peps/pep-0257/).

* **[pep8-naming](https://github.com/flintwork/pep8-naming)** is a naming convention checker for Python.

* **[flake8-debugger](https://github.com/JBKahn/flake8-debugger)** is a flake8 debug statement checker.

* **[flake8-import-order](https://github.com/public/flake8-import-order)** is a flake8 plugin that checks import order in the fashion of the Google Python Style Guide.

