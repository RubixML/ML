<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Other/Strategies/WildGuess.php">[source]</a></span>

# Wild Guess
Guess a random number somewhere between an upper and lower bound given by the data and a user-defined *shrinkage* parameter.

**Data Type:** Continuous

## Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | alpha | 0.5 | float | The range between the upper and lower bounds of the guess. A value of 1.0 indicates the full range of fitted values, whereas the range becomes narrower as the parameter goes to 0. |

## Additional Methods
Return the lower and upper bounds in a 2-tuple:
```php
public range() : array
```

## Example
```php
use Rubix\ML\Other\Strategies\WildGuess;

$strategy = new WildGuess(0.35);
```