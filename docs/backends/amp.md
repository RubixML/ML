<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Backends/Amp.php">[source]</a></span>

# Amp
[Amp Parallel](https://amphp.org/parallel/) is a multiprocessing subsystem that requires no extensions. It uses a non-blocking concurrency framework that implements coroutines using PHP generator functions under the hood.

!!! note
    The optimal number of workers will depend on the system specifications of the computer. Fewer workers than CPU cores may not achieve full processing potential but more workers than cores can cause excess overhead.

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | workers | Auto | int | The maximum number of workers in the worker pool. If null then tries to autodetect CPU core count. |

## Example
```php
use Rubix\ML\Backends\Amp;

$backend = new Amp(16);
```

## Additional Methods
Return the maximum number of workers in the worker pool:
```php
public workers() : int
```