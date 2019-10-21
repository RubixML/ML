# FAQ
Here you will find answers to the most frequently asked Rubix ML questions.

### What environment (SAPI) should I run Rubix ML in?
All Rubix ML programs are designed to run from the PHP [command line interface](http://php.net/manual/en/features.commandline.php) (CLI). The reason almost always boils down to performance and memory consumption.

If you want to serve your Rubix ML models in production then you can use the [Server](https://github.com/RubixML/Server) library to spin up an optimized standalone model server.

To run your script using the PHP command line interface (CLI), open a terminal window and enter:
```sh
$ php example.php
```

> **Note:** The PHP interpreter must be in your default PATH for the above syntax to work.

### I'm getting out of memory errors.
Try adjusting the `memory_limit` option in your php.ini file to something more reasonable. We recommend setting this to *-1* (no limit) or slightly below your device's memory supply for best results.

You can temporarily set the `memory_limit` in your script by using the `ini_set()` function.

**Example**

```php
ini_set('memory_limit', '-1');
```

> **Note:** Training can require a lot of memory. The amount necessary will depend on the amount of training data and the size of your model. If you have more data than you can hold in memory, some learners allow you to train them in batches. See the section on [Online](online.md) learners for more information.

### Training is slower than usual.
Training time depends on a number of factors including size of the dataset and complexity of the model. If you believe that training is taking unusually long then check the following factors.

- [Xdebug](https://xdebug.org/) or other debuggers are not enabled.
- You have enough RAM to hold the dataset and model in memory without swapping to disk.

### What is a Tuple?
A *tuple* is a way to denote an immutable sequential hetorogeneous list with a predefined length. An *n-tuple* is a tuple with the length of n. In some languages, tuples are a separate data type and their properties such as immutability are enforced by the compiler/interpreter. In PHP, tuples are denoted by sequential arrays which are mutable as a side effect.

**Example**

```php
$tuple = ['first', 'second', 0.001]; // a 3-tuple
```

### Does Rubix ML support multiprocessing?
Yes, Rubix supports parallel processing (multiprocessing) by utilizing a pluggable parallel computing [Backend](backends/api.md) under the hood. Objects that implement the [Parallel](parallel.md) interface are able to take advantage of parallel computing backends.

### Does Rubix ML support multithreading?
Not currently, however we plan to add CPU and GPU multithreading in the future.

### Does Rubix ML support Deep Learning?
Yes. A number of learners in Rubix ML support Deep Learning including the [Multi Layer Perceptron](classifiers/multi-layer-perceptron.md) classifier and [MLP Regressor](regressors/mlp-regressor.md).

### Does Rubix ML support Reinforcement Learning?
Not currently. Rubix ML is for supervised and unsupervised learning only.

### Does Rubix ML support time series data?
Yes and no. Currently, Rubix ML treats time series data like any other continuous feature. In the future, we may add algorithms that work specifically with a time component.