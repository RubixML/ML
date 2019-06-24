### Amp
[Amp](https://amphp.org/) Parallel is a multiprocessing subsystem that requires no extensions. It uses a non-blocking concurrency framework that implements coroutines using PHP generator functions under the hood.

> **Note**: If no worker count is given, the backend will try to auto detect the number of processor cores and set the worker count to that value. Autodetection works on most Windows and *nix-based systems.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Backends/Amp.php)

**Parameters:**

| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | workers | Auto | int | The maximum number of processes to execute in parallel. |

**Additional Methods:**

This backend does not have any additional methods.

**Example:**

```php
use Rubix\ML\Backends\Amp;

$backend = new Amp(); // Auto

$backend = new Amp(16);
```