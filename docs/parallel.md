### Parallel
Multiprocessing is the use of two or more processes that *usually* execute in parallel on multiple cores. Objects that implement the Parallel interface can take advantage of multi core systems by executing parts or all of their algorithm in parallel. Parallelizable objects can utilize various parallel processing [Backends](#backends) under the hood which are set using the `setBackend()` method.

> **Note**: The optimal number of workers will depend on the system specifications of the computer. Fewer workers than CPU cores may not achieve full processing potential but more workers than cores can cause excess overhead.

> **Note**: Unless otherwise stated, all objects implementing Parallel have a default backend of [Serial](#serial).

To set the backend processing engine:
```php
public setBackend(Backend $backend) : void
```

**Example:**

```php
use Rubix\ML\Backends\Amp;

$estimator->setBackend(new Amp(4));
```