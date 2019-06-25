<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Regressors/KDNeighborsRegressor.php">Source</a></span>

# K-d Neighbors Regressor
A fast implementation of [KNN Regressor](knn-regressor.md) using a spatially-aware K-d tree. The KDN Regressor works by locating the neighborhood of a sample via binary search and then does a brute force search only on the samples close to or within the neighborhood. The main advantage of K-d Neighbors over brute force KNN is inference speed, however you no longer have the ability to partially train.

**Interfaces:** [Estimator](../estimator.md), [Learner](../learner.md), [Persistable](../persistable.md)

**Data Type Compatibility:** Continuous

### Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | k | 3 | int | The number of neighboring training samples to consider when making a prediction. |
| 2 | kernel | Euclidean | object | The distance kernel used to compute the distance between sample points. |
| 3 | weighted | true | bool | Should we use the inverse distances as confidence scores when making predictions? |
| 4 | max leaf size | 30 | int | The max number of samples in a leaf node (*neighborhood*). |

### Additional Methods
Return the base k-d tree instance:
```php
public tree() : KDTree
```

### Example
```php
use Rubix\ML\Regressors\KDNeighborsRegressor;
use Rubix\ML\Kernels\Distance\Minkowski;

$estimator = new KDNeighborsRegressor(5, new Minkowski(4.0), true, 30);
```