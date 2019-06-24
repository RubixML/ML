<p><span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Regressors/KNNRegressor.php">Source</a></span></p>

# KNN Regressor
A version of [K Nearest Neighbors](#knn-regressor) that uses the average (mean) outcome of K nearest data points to make continuous valued predictions suitable for regression problems.

> **Note**: K Nearest Neighbors is considered a *lazy* learning estimator because it does the majority of its computation at prediction time.

**Interfaces:** [Estimator](#estimators), [Learner](#learner), [Online](#online), [Persistable](#persistable)

**Data Type Compatibility:** Continuous

### Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | k | 3 | int | The number of neighboring training samples to consider when making a prediction. |
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