<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Transformers/GaussianRandomProjector.php">[source]</a></span>

# Gaussian Random Projector
Random Projection is a dimensionality reduction technique based on the Johnson-Lindenstrauss lemma. It uses random matrices to project feature vectors onto a target number of dimensions. The Gaussian Random Projector utilizes a random matrix sampled from a smooth Gaussian distribution which projects samples onto a spherically random hyperplane through the origin.

**Interfaces:** [Transformer](api.md#transformer), [Stateful](api.md#stateful), [Persistable](../persistable.md)

**Data Type Compatibility:** Continuous only

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | dimensions | | int | The number of target dimensions to project onto. |

## Example
```php
use Rubix\ML\Transformers\GaussianRandomProjector;

$transformer = new GaussianRandomProjector(100);
```

## Additional Methods
Estimate the minimum dimensionality needed to satisfy a *max distortion* constraint with *n* samples using the Johnson-Lindenstrauss lemma:
```php
public static minDimensions(int $n, float $maxDistortion = 0.5) : int
```

```php
use Rubix\ML\Transformers\GaussianRandomProjector;

$dimensions = GaussianRandomProjector::minDimensions(5000, 0.2);
```