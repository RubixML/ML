<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Transformers/SparseRandomProjector.php">[source]</a></span>

# Sparse Random Projector
A *database-friendly* random projector that samples its random projection matrix from a sparse probabilistic approximation of the Gaussian distribution. The term *database-friendly* refers to the fact that the number of non-zero operations required to transform the input matrix is reduced by the sparsity factor.

**Interfaces:** [Transformer](api.md#transformer), [Stateful](api.md#stateful), [Persistable](../persistable.md)

**Data Type Compatibility:** Continuous only

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | dimensions | | int | The number of target dimensions to project onto. |
| 2 | sparsity | 0.66 | float | The proportion of zero to non-zero elements in the random projection matrix. If null, sparsity factor will be chosen automatically. |

## Example
```php
use Rubix\ML\Transformers\SparseRandomProjector;

$transformer = new SparseRandomProjector(30, null);
```

## Additional Methods
Estimate the minimum dimensionality needed to satisfy a *max distortion* constraint with *n* samples using the Johnson-Lindenstrauss lemma:
```php
public static minDimensions(int $n, float $maxDistortion = 0.5) : int
```

```php
use Rubix\ML\Transformers\SparseRandomProjector;

$dimensions = SparseRandomProjector::minDimensions(10000, 0.5);
```

## References
[^1]: D. Achlioptas. (2003). Database-friendly random projections: Johnson-Lindenstrauss with binary coins.
[^2]: P. Li at al. (2006). Very Sparse Random Projections.