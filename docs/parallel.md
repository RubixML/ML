# Parallel
Multiprocessing is the use of two or more processes that *usually* execute in parallel on multiple cores. Objects that implement the Parallel interface can take advantage of multicore systems by executing parts or all of their algorithm in parallel. Parallelizable objects can utilize various parallel processing Backends under the hood which are set using the `setBackend()` method.

> **Note:** Most parallel learners are configured to use a [Serial](backends/serial.md) backend by default.

### Set Backend
To set the backend processing engine:
```php
public setBackend(Backend $backend) : void
```

**Example**

```php
use Rubix\ML\Backends\Amp;

$estimator->setBackend(new Amp(16));
```
