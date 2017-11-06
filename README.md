# DNC-TensorFlow
An implementation of the Differential Neural Computer (DNC) in [TensorFlow](https://www.tensorflow.org/), using [Sonnet](https://github.com/deepmind/sonnet), introduced in DeepMind's Nature paper:
> [“Hybrid computing using a neural network with dynamic external memory", Nature 538, 471–476 (October 2016) doi:10.1038/nature20101.](http://www.nature.com/articles/nature20101.epdf?author_access_token=ImTXBI8aWbYxYQ51Plys8NRgN0jAjWel9jnR3ZoTv0MggmpDmwljGswxVdeocYSurJ3hxupzWuRNeGvvXnoO8o4jTJcnAyhGuZzXJ1GEaD-Z7E6X_a9R-xqJ9TfJWBqz)

**NOTE**: this is currently a work-in-progress. The DNC model is now complete, but tasks to test the DNC learning capabilities are still in progress.

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

