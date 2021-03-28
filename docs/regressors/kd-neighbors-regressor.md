<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Regressors/KDNeighborsRegressor.php">[source]</a></span>

# K-d Neighbors Regressor
A fast implementation of [KNN Regressor](knn-regressor.md) using a spatially-aware binary tree for nearest neighbors search. K-d Neighbors Regressor works by locating the neighborhood of a sample via binary search and then does a brute force search only on the samples close to or within the neighborhood of the unknown sample. The main advantage of K-d Neighbors over brute force KNN is inference speed, however, it cannot be partially trained.

**Interfaces:** [Estimator](../estimator.md), [Learner](../learner.md), [Persistable](../persistable.md)

**Data Type Compatibility:** Depends on distance kernel

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | k | 5 | int | The number of nearest neighbors to consider when making a prediction. |
| 2 | weighted | false | bool | Should we consider the distances of our nearest neighbors when making predictions? |
| 3 | tree | KDTree | Spatial | The spatial tree used to run nearest neighbor searches. |

## Example
```php
use Rubix\ML\Regressors\KDNeighborsRegressor;
use Rubix\ML\Graph\Trees\BallTree;

$estimator = new KDNeighborsRegressor(20, true, new BallTree(50));
```

## Additional Methods
Return the base spatial tree instance:
```php
public tree() : Spatial
```
