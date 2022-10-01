<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Transformers/HotDeckImputer.php">[source]</a></span>

# Hot Deck Imputer
A *hot deck* is a set of complete donor samples that may be referenced when imputing a value for a missing feature value. Hot Deck Imputer first finds the k most similar donors to a sample that contains a missing value and then chooses a value at random from those donors.

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
use Rubix\ML\Transformers\HotDeckImputer;
use Rubix\ML\Graph\Trees\BallTree;
use Rubix\ML\Kernels\Distance\Gower;

$transformer = new HotDeckImputer(20, false, '?', new BallTree(50, new Gower(1.0)));
```

## Additional Methods
This transformer does not have any additional methods.

## References
[^1]: C. Hasler et al. (2015). Balanced k-Nearest Neighbor Imputation.
