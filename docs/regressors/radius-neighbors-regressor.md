<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Regressors/RadiusNeighborsRegressor.php">Source</a></span>

# Radius Neighbors Regressor
This is the regressor version of [Radius Neighbors](../classifiers/radius-neighbors.md) implementing a binary spatial tree under the hood for fast radius queries. The prediction is a weighted average of each label from the training set that is within a fixed user-defined radius.

> **Note**: Unknown samples with no training samples within radius are labeled *NaN*. As such, Radius Neighbors is also a quasi anomaly detector.

**Interfaces:** [Estimator](../estimator.md), [Learner](../learner.md), [Persistable](../persistable.md)

**Data Type Compatibility:** Continuous

### Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | radius | 1.0 | float | The radius within which points are considered neighboors. |
| 2 | weighted | true | bool | Should we use the inverse distances as confidence scores when making predictions? |
| 3 | tree | BallTree | object | The spatial tree used to run range searches. |

### Additional Methods
Return the base spatial tree instance:
```php
public tree() : Spatial
```

### Example
```php
use Rubix\ML\Regressors\RadiusNeighborsRegressor;
use Rubix\ML\Graph\Trees\BallTree;
use Rubix\ML\Kernels\Distance\Diagonal;

$estimator = new RadiusNeighborsRegressor(0.5, true, new BallTree(30, new Diagonal()));
```