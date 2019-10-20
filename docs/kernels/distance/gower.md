<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Kernels/Distance/Gower.php">Source</a></span>

# Gower
A robust distance kernel that measures a mix of categorical and continuous data types while handling NaN values. When comparing continuous data, the Gower metric is equivalent to the normalized Manhattan distance and when comparing categorical data it is equivalent to the Hamming distance.

> **Note:** The Gower metric expects that all continuous variables are on the same scale. By default, the range is between 0 and 1.

**Data Type Compatibility:** Continuous, Categorical

### Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | range | 1.0 | float | The range of the continuous feature columns. |

### Example
```php
use Rubix\ML\Kernels\Distance\Gower;

$kernel = new Gower(); // Continuous features between 0 and 1

$kernel = new Gower(2.0); // Between -1 and 1

$kernel = new Gower(1000.0); // Between 0 and 1000
```

### References
>- J. C. Gower. (1971). A General Coefficient of Similarity and Some of Its Properties.