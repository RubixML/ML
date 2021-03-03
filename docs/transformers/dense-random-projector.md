<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Transformers/DenseRandomProjector.php">[source]</a></span>

# Dense Random Projector
A *database-friendly* random projector with projection matrix sampled from a dense uniform distribution ([-1, 1]).

!!! note
    Dense Random Projector has been deprecated, use [Sparse Random Projector](sparse-random-projector.md) with sparsity set to 0 instead.

**Interfaces:** [Transformer](api.md#transformer), [Stateful](api.md#stateful), [Persistable](../persistable.md)

**Data Type Compatibility:** Continuous only

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | dimensions | | int | The number of target dimensions to project onto. |

## Example
```php
use Rubix\ML\Transformers\DenseRandomProjector;

$transformer = new DenseRandomProjector(50);
```

## Additional Methods
Estimate the minimum dimensionality needed to satisfy a *max distortion* constraint with *n* samples using the Johnson-Lindenstrauss lemma:
```php
public static minDimensions(int $n, float $maxDistortion = 0.5) : int
```

```php
use Rubix\ML\Transformers\DenseRandomProjector;

$dimensions = DenseRandomProjector::minDimensions(1000, 0.3);
```

## References
[^1]: D. Achlioptas. (2003). Database-friendly random projections: Johnson-Lindenstrauss with binary coins.