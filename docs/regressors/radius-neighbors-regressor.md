<p><span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Regressors/RadiusNeighborsRegressor.php">Source</a></span></p>

# Radius Neighbors Regressor
This is the regressor version of [Radius Neighbors](#radius-neighbors) classifier implementing a binary spatial tree under the hood for fast radius queries. The prediction is a weighted average of each label from the training set that is within a fixed user-defined radius.

> **Note**: Unknown samples with 0 samples from the training set that are within radius will be labeled *NaN*.

**Interfaces:** [Estimator](#estimators), [Learner](#learner), [Persistable](#persistable)

**Data Type Compatibility:** Continuous

### Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | radius | 1.0 | float | The radius within which points are considered neighboors. |
| 2 | kernel | Euclidean | object | The distance kernel used to compute the distance between sample points. |
| 3 | weighted | true | bool | Should we use the inverse distances as confidence scores when making predictions? |
| 4 | max leaf size | 30 | int | The max number of samples in a leaf node (*ball*). |

### Additional Methods
Return the height of the tree:
```php
public height() : int
```

Return the balance of the tree:
```php
public balance() : int
```

### Example
```php
use Rubix\ML\Regressors\RadiusNeighborsRegressor;
use Rubix\ML\Kernels\Distance\Diagonal;

$estimator = new RadiusNeighborsRegressor(0.5, new Diagonal(), true, 20);
```