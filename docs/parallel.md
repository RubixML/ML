# Parallel
Multiprocessing is the use of two or more processes that execute in parallel. Objects that implement the Parallel interface can take advantage of multicore processors by executing parts or all of the algorithm in parallel. Choose a number of processes equal to the number of CPU cores in order to take advantage of a system's full processing capability.

!!! note
    Most parallel learners are configured to use the [Serial](backends/serial.md) backend by default.

## Set a Backend
Parallelizable objects can utilize a parallel processing Backend by passing it to the `setBackend()` method.

To set the backend processing engine:
```php
public setBackend(Backend $backend) : void
```

```php
use Rubix\ML\Classifiers\RandomForest;
use Rubix\ML\Backends\Amp;

$estimator = new RandomForest();

$estimator->setBackend(new Amp(16));
```
