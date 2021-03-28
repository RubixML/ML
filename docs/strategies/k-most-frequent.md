<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Strategies/KMostFrequent.php">[source]</a></span>

# K Most Frequent
This Strategy outputs one of k most frequently occurring classes at random with equal probability.

**Data Type:** Categorical

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | k | 1 | int | The number of most frequent classes to consider. |

## Example
```php
use Rubix\ML\Strategies\KMostFrequent;

$strategy = new KMostFrequent(5);
```