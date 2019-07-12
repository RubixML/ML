<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Classifiers/KDNeighbors.php">Source</a></span>

# K-d Neighbors
A fast [K Nearest Neighbors](k-nearest-neighbors.md) algorithm that uses a K-d tree to divide the training set into neighborhoods whose max size are controlled by the max leaf size parameter. K-d Neighbors does a binary search to locate the nearest neighborhood and then prunes all neighborhoods whose bounding box is further than the kth nearest neighbor found so far. The main advantage of K-d Neighbors over regular brute force KNN is that it is faster, however it cannot be partially trained.

**Interfaces:** [Estimator](../estimator.md), [Learner](../learner.md), [Probabilistic](../probabilistic.md), [Persistable](../persistable.md)

**Data Type Compatibility:** Continuous

### Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | k | 5 | int | The number of nearest neighbors to consider when making a prediction. |
| 2 | tree | KDTree | object | The spatial tree used for nearest neighbor queries. |
| 3 | weighted | true | bool | Should we use the inverse distances as confidence scores when making predictions? |

### Additional Methods
Return the base spatial tree instance:
```php
public tree() : Spatial
```

### Example
```php
use Rubix\ML\Classifiers\KDNeighbors;
use Rubix\ML\Graph\Trees\BallTree;
use Rubix\ML\Kernels\Distance\Minskowski;

$estimator = new KDNeighbors(3, new BallTree(40, new Minkowski()), false);
```