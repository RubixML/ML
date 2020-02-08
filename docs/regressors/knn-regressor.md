<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Regressors/KNNRegressor.php">[source]</a></span>

# KNN Regressor
A version of the K Nearest Neighbors algorithm that uses the average (mean) outcome of the *k* nearest data points to an unknown sample to make continuous-valued predictions suitable for regression problems.

> **Note:** KNN is considered a *lazy* learner because it does the majority of its computation during inference. For a faster spatial tree-accelerated version, see [KD Neighbors Regressor](kd-neighbors-regressor.md).

**Interfaces:** [Estimator](../estimator.md), [Learner](../learner.md), [Online](../online.md), [Persistable](../persistable.md)

**Data Type Compatibility:** Depends on distance kernel

## Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | k | 5 | int | The number of nearest neighbors to consider when making a prediction. |
| 2 | weighted | true | bool | Should we consider the distances of our nearest neighbors when making predictions? |
| 3 | kernel | Euclidean | Distance | The distance kernel used to compute the distance between sample points. |

## Additional Methods
This estimator does not have any additional methods.

## Example
```php
use Rubix\ML\Regressors\KNNRegressor;
use Rubix\ML\Kernels\Distance\SafeEuclidean;

$estimator = new KNNRegressor(2, false, new SafeEuclidean());
```