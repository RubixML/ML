<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Regressors/RadiusNeighborsRegressor.php">[source]</a></span>

# Radius Neighbors Regressor
This is the regressor version of [Radius Neighbors](../classifiers/radius-neighbors.md) implementing a binary spatial tree under the hood for fast radius queries. The prediction is a weighted average of each label from the training set that is within a fixed user-defined radius.

> **Note**: Samples with 0 neighbors within radius will be predicted *NaN*.

**Interfaces:** [Estimator](../estimator.md), [Learner](../learner.md), [Persistable](../persistable.md)

**Data Type Compatibility:** Depends on distance kernel

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | radius | 1.0 | float | The radius within which points are considered neighbors. |
| 2 | weighted | true | bool | Should we consider the distances of our nearest neighbors when making predictions? |
| 3 | tree | BallTree | Spatial | The spatial tree used to run range searches. |

## Example
```php
use Rubix\ML\Regressors\RadiusNeighborsRegressor;
use Rubix\ML\Graph\Trees\BallTree;
use Rubix\ML\Kernels\Distance\Diagonal;

$estimator = new RadiusNeighborsRegressor(0.5, true, new BallTree(30, new Diagonal()));
```

## Additional Methods
Return the base spatial tree instance:
```php
public tree() : Spatial
```
