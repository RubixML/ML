<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Transformers/GaussianRandomProjector.php">[source]</a></span>

# Gaussian Random Projector
A random projector is a dimensionality reducer based on the Johnson-Lindenstrauss lemma that uses a random matrix to project feature vectors onto a user-specified number of dimensions. It is faster than most non-randomized dimensionality reduction techniques such as [PCA](principal-component-analysis.md) or [LDA](linear-discriminant-analysis.md) and it offers similar results. This version utilizes a random matrix sampled from a smooth Gaussian distribution.

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
use Rubix\ML\Transformers\GaussianRandomProjector;

$dimensions = GaussianRandomProjector::estimate(2000);

$transformer = new GaussianRandomProjector($dimensions); // Using estimate

$transformer = new GaussianRandomProjector(100); // Specifying dimensionality
```