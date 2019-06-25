<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Classifiers/RadiusNeighbors.php">Source</a></span>

# Radius Neighbors
Radius Neighbors is a spatial tree-based classifier that takes the weighted vote of each neighbor within a fixed user-defined radius measured by a kernel distance function.

> **Note:** Unknown samples with 0 samples from the training set that are within radius will be labeled as outliers (*-1*).

**Interfaces:** [Estimator](../estimator.md), [Learner](../learner.md), [Probabilistic](../probabilistic.md), [Persistable](../persistable.md)

**Data Type Compatibility:** Continuous

### Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | radius | 1.0 | float | The radius within which points are considered neighboors. |
| 2 | kernel | Euclidean | object | The distance kernel used to compute the distance between sample points. |
| 3 | weighted | true | bool | Should we use the inverse distances as confidence scores when making predictions? |
| 4 | max leaf size | 30 | int | The max number of samples in a leaf node (*ball*). |

### Additional Methods
Return the base ball tree instance:
```php
public tree() : BallTree
```

### Example
```php
use Rubix\ML\Classifiers\RadiusNeighbors;
use Rubix\ML\Kernels\Distance\Manhattan;

$estimator = new RadiusNeighbors(50.0, new Manhattan(), false, 30);
```