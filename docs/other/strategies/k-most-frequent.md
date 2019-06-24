<p><span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Other/Strategies/KMostFrequent.php">Source</a></span></p>

# K Most Frequent
This strategy outputs one of K most frequent discrete values at random.

**Data Type:** Categorical

### Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | k | 1 | int | The number of most frequency categories to consider when formulating a guess. |

### Example
```php
use Rubix\ML\Other\Strategies\KMostFrequent;

$strategy = new KMostFrequent(5);
```