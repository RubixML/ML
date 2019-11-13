<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Classifiers/KNearestNeighbors.php">[source]</a></span>

# K Nearest Neighbors
A distance-based algorithm that locates the k nearest neighbors (data points) from the training set and uses a weighted vote to classify unknown samples during inference. A *kernelized* distance function allows the user to specify different concepts of distance to the estimator. 

> **Note:** This learner is considered a *lazy* learner because it does the majority of its computation during inference. For a fast spatial tree-accelerated version, see [KD Neighbors](kd-neighbors.md).

**Interfaces:** [Estimator](../estimator.md), [Learner](../learner.md), [Online](../online.md), [Probabilistic](../probabilistic.md), [Persistable](../persistable.md)

**Data Type Compatibility:** Depends on the distance kernel

### Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | k | 5 | int | The number of nearest neighbors to consider when making a prediction. |
| 2 | weighted | true | bool | Should we use the inverse distances as confidence scores when making predictions? |
| 3 | kernel | Euclidean | object | The distance kernel used to compute the distance between sample points. |

### Additional Methods
This estimator does not have any additional methods.

### Example
```php
use Rubix\ML\Classifiers\KNearestNeighbors;
use Rubix\ML\Kernels\Distance\Manhattan;

$estimator = new KNearestNeighbors(3, true, new Manhattan());
```