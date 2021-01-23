<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Other/Strategies/Constant.php">[source]</a></span>'

# Constant
Always guess the same value.

**Data Type:** Continuous

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | value | 0.0 | float | The value to constantly guess. |

## Example
```php
use Rubix\ML\Other\Strategies\Constant;

$strategy = new Constant(0.0);
```