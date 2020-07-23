<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Transformers/SparseRandomProjector.php">[source]</a></span>

# Sparse Random Projector
The Sparse Random Projector uses a random matrix sampled from a sparse Gaussian distribution (mostly *0*s) to reduce the dimensionality of a dataset.

**Interfaces:** [Transformer](api.md#transformer), [Stateful](api.md#stateful)

**Data Type Compatibility:** Continuous only

## Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | dimensions | | int | The number of target dimensions to project onto. |

## Example
```php
use Rubix\ML\Transformers\SparseRandomProjector;

$transformer = new SparseRandomProjector(30);
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

### References
>- D. Achlioptas. (2003). Database-friendly random projections: Johnson-Lindenstrauss with binary coins.