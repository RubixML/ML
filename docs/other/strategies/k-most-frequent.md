<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Other/Strategies/KMostFrequent.php">Source</a></span>

# K Most Frequent
This Strategy outputs one of k most frequently occurring classes at random with equal probability.

**Data Type:** Categorical

### Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | k | 1 | int | The number of most frequent classes to consider. |

### Additional Methods
Return the k most frequent classes:
```php
public classes() : array
```

### Example
```php
use Rubix\ML\Other\Strategies\KMostFrequent;

$strategy = new KMostFrequent(5);
```