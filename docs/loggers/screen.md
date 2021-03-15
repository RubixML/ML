<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Loggers/Screen.php">[source]</a></span>

# Screen
A logger that displays log messages to the standard output.

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | channel | '' | string | The channel name that appears on each line. |
| 2 | timestampFormat | 'Y-m-d H:i:s' | string | The format of the timestamp. |

## Example
```php
use Rubix\ML\Loggers\Screen;

$logger = new Screen('mlp', 'Y-m-d H:i:s');
```
