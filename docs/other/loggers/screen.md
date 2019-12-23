# Screen
A simple logger that outputs directly to the console.

## Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | channel | 'main' | string | The channel name that appears on each line. |
| 2 | format | 'Y-m-d H:i:s' | string | The format of the timestamp. |

## Example
```php
use Rubix\ML\Other\Loggers\Screen;

$logger = new Screen('credit', 'Y-m-d H:i:s');
```