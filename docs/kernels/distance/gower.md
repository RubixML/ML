<span style="float:right;"><a href="https://github.com/RubixML/Extras/blob/master/src/Kernels/Distance/Gower.php">[source]</a></span>

# Gower
A robust distance kernel that measures samples consisting of a mix of categorical and continuous data types while also handling missing (NaN) values. When comparing continuous data, the Gower metric is equivalent to the normalized [Manhattan](manhattan.md) distance and when comparing categorical data it is equivalent to the [Hamming](hamming.md) distance.

> **Note:** The Gower metric expects all continuous variables to have a standardized range. The default range works for values that have been normalized between 0 and 1.

**Data Type Compatibility:** Continuous, Categorical

## Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | range | 1.0 | float | The standardized range of the continuous feature columns. Ex. [0, 1] has a range of 1, [-1, 1] has a range of 2, and so forth. |

## Example
```php
use Rubix\ML\Kernels\Distance\Gower;

$kernel = new Gower(2.0);
```

### References
>- J. C. Gower. (1971). A General Coefficient of Similarity and Some of Its Properties.