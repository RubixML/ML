<p><span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Classifiers/KDNeighbors.php">Source</a></span></p>

# K-d Neighbors
A fast [K Nearest Neighbors](#k-nearest-neighbors) algorithm that uses a K-d tree to divide the training set into neighborhoods whose max size are controlled by the max leaf size parameter. K-d Neighbors does a binary search to locate the nearest neighborhood and then prunes all neighborhoods whose bounding box is further than the kth nearest neighbor found so far. The main advantage of K-d Neighbors over regular brute force KNN is that it is faster, however it cannot be partially trained.

**Interfaces:** [Estimator](#estimators), [Learner](#learner), [Probabilistic](#probabilistic), [Persistable](#persistable)

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
use Rubix\ML\Classifiers\KDNeighbors;
use Rubix\ML\Kernels\Distance\Euclidean;

$estimator = new KDNeighbors(3, new Euclidean(), false, 10);
```