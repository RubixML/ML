<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Backends/Amp.php">[source]</a></span>

# Amp
[Amp Parallel](https://amphp.org/parallel/) is a multiprocessing subsystem that requires no extensions. It uses a non-blocking concurrency framework that implements coroutines using PHP generator functions under the hood.

> **Note:** The optimal number of workers will depend on the system specifications of the computer. Fewer workers than CPU cores may not achieve full processing potential but more workers than cores can cause excess overhead.

## Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | workers | Auto | int | The maximum number of workers i.e. processes to execute in parallel. |

## Additional Methods
Return the maximum number of workers:
```php
public workers() : int
```

## Example
```php
use Rubix\ML\Backends\Amp;

$backend = new Amp(16);
```