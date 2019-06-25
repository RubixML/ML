<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Backends/Amp.php">Source</a></span>

# Amp
[Amp Parallel](https://amphp.org/parallel/) is a multiprocessing subsystem that requires no extensions. It uses a non-blocking concurrency framework that implements coroutines using PHP generator functions under the hood.

> **Note:** If no worker count is given, the backend will try to auto detect the number of processor cores and set the worker count to that value. Autodetection works on most Windows and *nix-based systems.

> **Note:** The optimal number of workers will depend on the system specifications of the computer. Fewer workers than CPU cores may not achieve full processing potential but more workers than cores can cause excess overhead.

### Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | workers | Auto | int | The maximum number of processes to execute in parallel. |

### Additional Methods
This backend does not have any additional methods.

### Example
```php
use Rubix\ML\Backends\Amp;

$backend = new Amp(); // Autodetect workers

$backend = new Amp(16);
```