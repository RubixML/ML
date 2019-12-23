<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Transformers/DenseRandomProjector.php">[source]</a></span>

# Dense Random Projector
The Dense Random Projector uses a random matrix sampled from a dense uniform distribution [-1, 1] to reduce the dimensionality of a dataset by projecting it onto a vector space of target dimensionality.

**Interfaces:** [Transformer](api.md#transformer), [Stateful](api.md#stateful)

**Data Type Compatibility:** Continuous only

## Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | dimensions | | int | The number of target dimensions to project onto. |

## Additional Methods
Estimate the minimum dimensionality needed given total sample size and *max distortion* using the Johnson-Lindenstrauss lemma:
```php
public static estimate(int $n, float $maxDistortion = 0.1) : int
```

## Example
```php
use Rubix\ML\Transformers\DenseRandomProjector;

$dimensions = DenseRandomProjector::estimate(3500);

$transformer = new DenseRandomProjector($dimensions); // Using estimate

$transformer = new DenseRandomProjector(50); // Specifying dimensionality
```

### References
>- D. Achlioptas. (2003). Database-friendly random projections: Johnson-Lindenstrauss with binary coins.