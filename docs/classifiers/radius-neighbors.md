<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Classifiers/RadiusNeighbors.php">[source]</a></span>

# Radius Neighbors
Radius Neighbors is a classifier that takes the distance-weighted vote of each neighbor within a cluster of a fixed user-defined radius to make a prediction. Since the radius of the search can be constrained, Radius Neighbors is more robust to outliers than [K Nearest Neighbors](k-nearest-neighbors.md). In addition, Radius Neighbors acts as a quasi-anomaly detector by flagging samples that have 0 neighbors within the search radius.

**Interfaces:** [Estimator](../estimator.md), [Learner](../learner.md), [Probabilistic](../probabilistic.md), [Persistable](../persistable.md)

**Data Type Compatibility:** Depends on distance kernel

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | radius | 1.0 | float | The radius within which points are considered neighbors. |
| 2 | weighted | false | bool | Should we consider the distances of our nearest neighbors when making predictions? |
| 3 | outlierClass | '?' | string | The class label for any samples that have 0 neighbors within the specified radius. |
| 4 | tree | BallTree | Spatial | The spatial tree used to run range searches. |

## Example
```php
use Rubix\ML\Classifiers\RadiusNeighbors;
use Rubix\ML\Graph\Trees\KDTree;
use Rubix\ML\Kernels\Distance\Manhattan;

$estimator = new RadiusNeighbors(50.0, true, '?', new KDTree(100, new Manhattan()));
```

## Additional Methods
Return the base spatial tree instance:
```php
public tree() : Spatial
```
