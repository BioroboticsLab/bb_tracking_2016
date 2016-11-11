# bb_tracking

[![Build Status](https://secure.travis-ci.org/BioroboticsLab/bb_tracking.svg?branch=master)](http://travis-ci.org/BioroboticsLab/bb_tracking?branch=master)
[![Coverage Status](https://coveralls.io/repos/github/BioroboticsLab/bb_tracking/badge.svg?branch=master)](https://coveralls.io/github/BioroboticsLab/bb_tracking?branch=master)
[![Documentation Status](https://readthedocs.org/projects/bb-tracking/badge/?version=latest)](http://bb-tracking.readthedocs.io/en/latest/?badge=latest)

Python code to perform tracking and evaluate the performance of tracking algorithms for the beesbook project.

## Python Interface

To install the python interface simply run:

```
$ pip install git+https://github.com/BioroboticsLab/bb_tracking.git
```

Developers use the following setup:

First clone the repository, then enter the directory and run:

```
$ pip install -e .[develop]
```

Depending on your python environment you might also have to run:

```
$ pip install -r requirements-dev.txt
```

## Documentation and Tutorials

Check the [documentation](http://bb-tracking.readthedocs.io/en/latest/) and the [Getting Started with bb_tracking](https://github.com/BioroboticsLab/bb_tracking/blob/master/getting-started-with-bb-tracking.ipynb) notebook.

Note that you will need [IPython](https://ipython.org/) and [Jupyter](http://jupyter.org/) for the [Getting Started]((https://github.com/BioroboticsLab/bb_tracking/blob/master/getting-started-with-bb-tracking.ipynb)) notebook.
They are **not** part of this module and you will have to install them by yourself.
