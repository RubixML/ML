<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Strategies/Percentile.php">[source]</a></span>

# Blurry Percentile
A strategy that always guesses the p-th percentile of the fitted data.

**Data Type:** Continuous

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | p | 50.0 | float | The percentile of the fitted data to use as a guess. |

## Example
```php
use Rubix\ML\Strategies\Percentile;

$strategy = new Percentile(90.0);
```