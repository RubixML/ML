<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Regressors/KNNRegressor.php">Source</a></span>

# KNN Regressor
A version of the K Nearest Neighbors algorithm that uses the average (mean) outcome of the k nearest data points to make continuous valued predictions suitable for regression problems.

> **Note:** This learner is considered a *lazy* learner because it does the majority of its computation during inference. For a fast spatial tree-based version, see [KD Neighbors Regressor](kd-neighbors-regressor.md).

**Interfaces:** [Estimator](../estimator.md), [Learner](../learner.md), [Online](../online.md), [Persistable](../persistable.md)

**Data Type Compatibility:** Continuous

### Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | k | 5 | int | The number of nearest neighbors to consider when making a prediction. |
| 2 | kernel | Euclidean | object | The distance kernel used to compute the distance between sample points. |
| 3 | weighted | true | bool | Should we use the inverse distances as confidence scores when making predictions? |

### Additional Methods
This estimator does not have any additional methods.

### Example
```php
use Rubix\ML\Regressors\KNNRegressor;
use Rubix\ML\Kernels\Distance\Minkowski;

$estimator = new KNNRegressor(2, new Minkowski(3.0), false);
```