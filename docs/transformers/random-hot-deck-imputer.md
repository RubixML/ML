<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Transformers/RandomHotDeckImputer.php">[source]</a></span>

# Random Hot Deck Imputer
A *hot deck* is a set of complete donor samples that may be referenced when imputing a value for a missing feature value. Random Hot Deck Imputer first finds the k nearest donors to a sample that contains a missing value and then chooses a value at random from the neighborhood.

!!! note
    Requires a NaN safe distance kernel such as [Safe Euclidean](../kernels/distance/safe-euclidean.md) for continuous features.

**Interfaces:** [Transformer](api.md#transformers), [Stateful](api.md#stateful), [Persistable](../persistable.md)

**Data Type Compatibility:** Depends on distance kernel

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | k | 5 | int | The number of nearest neighbor donors to consider when imputing a value. |
| 2 | weighted | true | bool | Should we use distances as weights when selecting a donor sample? |
| 3 | categoricalPlaceholder | '?' | string | The categorical placeholder denoting the category that contains missing values. |
| 4 | tree | BallTree | Spatial | The spatial tree used to run nearest neighbor searches. |

## Example
```php
use Rubix\ML\Transformers\RandomHotDeckImputer;
use Rubix\ML\Graph\Trees\BallTree;
use Rubix\ML\Kernels\Distance\SafeEuclidean;

$transformer = new RandomHotDeckImputer(20, true, '?', new BallTree(50, new SafeEuclidean()));
```

## Additional Methods
This transformer does not have any additional methods.

## References
[^1]: C. Hasler et al. (2015). Balanced k-Nearest Neighbor Imputation.