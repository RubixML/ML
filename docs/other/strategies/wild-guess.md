<p><span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Other/Strategies/WildGuess.php">Source</a></span></p>

# Wild Guess
It is what you think it is. Make a guess somewhere in between the minimum and maximum values observed during fitting with equal probability given to all values within range.

**Data Type:** Continuous

### Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | precision | 2 | int | The number of decimal places of precision for each guess. |

### Example
```php
use Rubix\ML\Other\Strategies\WildGuess;

$strategy = new WildGuess(5);
```