<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Transformers/KNNImputer.php">[source]</a></span>

# KNN Imputer
An unsupervised imputer that replaces missing values in datasets with the distance-weighted average of the samples' *k* nearest neighbors' values.

**Note:** NaN safe distance kernels, such as [Safe Euclidean](../kernels/distance/safe-euclidean.md), are required for continuous features.

**Interfaces:** [Transformer](api.md#transformers), [Stateful](api.md#stateful)

**Data Type Compatibility:** Depends on distance kernel

## Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | k | 5 | int | The number of nearest neighbors to consider when imputing a value. |
| 2 | weighted | true | bool | Should we use the inverse distances as confidence scores when imputing values? |
| 3 | placeholder | '?' | string | The categorical placeholder denoting the category that contains missing values. |
| 4 | tree | BallTree | Spatial | The spatial tree used to run nearest neighbor searches. |

## Additional Methods
This transformer does not have any additional methods.

## Example
```php
use Rubix\ML\Transformers\KNNImputer;
use Rubix\ML\Graph\Trees\BallTee;
use Rubix\ML\Kernels\Distance\SafeEuclidean;

$transformer = new KNNImputer(10, false, '?', new BallTree(30, new SafeEuclidean()));
```

### References
>- O. Troyanskaya et al. (2001). Missing value estimation methods for DNA microarrays.