<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Transformers/KNNImputer.php">[source]</a></span>

# KNN Imputer
An unsupervised imputer that replaces missing values in a dataset with the distance-weighted average of the samples' *k* nearest neighbors' values. The average for a continuous feature column is defined as the mean of the values of each donor. Similarly, average is defined as the *most frequent* value for categorical features.

!!! note
    Requires a NaN safe distance kernel such as [Safe Euclidean](../kernels/distance/safe-euclidean.md) for continuous features.

**Interfaces:** [Transformer](api.md#transformers), [Stateful](api.md#stateful), [Persistable](../persistable.md)

**Data Type Compatibility:** Depends on distance kernel

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | k | 5 | int | The number of nearest neighbor donors to consider when imputing a value. |
| 2 | weighted | false | bool | Should we use distances as weights when selecting a donor sample? |
| 3 | categoricalPlaceholder | '?' | string | The categorical placeholder denoting the category that contains missing values. |
| 4 | tree | BallTree | Spatial | The spatial tree used to run nearest neighbor searches. |

## Example
```php
use Rubix\ML\Transformers\KNNImputer;
use Rubix\ML\Graph\Trees\BallTee;
use Rubix\ML\Kernels\Distance\SafeEuclidean;

$transformer = new KNNImputer(10, false, '?', new BallTree(30, new SafeEuclidean()));
```

## Additional Methods
This transformer does not have any additional methods.

## References
[^1]: O. Troyanskaya et al. (2001). Missing value estimation methods for DNA microarrays.
