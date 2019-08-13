<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Regressors/KDNeighborsRegressor.php">Source</a></span>

# K-d Neighbors Regressor
A fast implementation of [KNN Regressor](knn-regressor.md) using a spatially-aware binary tree. The KDN Regressor works by locating the neighborhood of a sample via binary search and then does a brute force search only on the samples close to or within the neighborhood of the unknown sample. The main advantage of K-d Neighbors over brute force KNN is inference speed, however you no longer have the ability to partially train.

**Interfaces:** [Estimator](../estimator.md), [Learner](../learner.md), [Persistable](../persistable.md)

**Data Type Compatibility:** Continuous

### Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | k | 5 | int | The number of nearest neighbors to consider when making a prediction. |
| 2 | weighted | true | bool | Should we use the inverse distances as confidence scores when making predictions? |
| 3 | tree | KDTree | object | The spatial tree used for nearest neighbor queries. |

### Additional Methods
Return the base spatial tree instance:
```php
public tree() : Spatial
```

### Example
```php
use Rubix\ML\Regressors\KDNeighborsRegressor;
use Rubix\ML\Graph\Trees\BallTree;

$estimator = new KDNeighborsRegressor(5, true, new BallTree(50));
```