<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Classifiers/KNearestNeighbors.php">[source]</a></span>

# K Nearest Neighbors
A brute-force distance-based learning algorithm that locates the *k* nearest samples from the training set and predicts the class label that is most common. K Nearest Neighbors (KNN) is considered a *lazy* learner because it performs most of its computation at inference time.

!!! note
    For a faster spatial tree-accelerated version of KNN, see [KD Neighbors](kd-neighbors.md).

**Interfaces:** [Estimator](../estimator.md), [Learner](../learner.md), [Online](../online.md), [Probabilistic](../probabilistic.md), [Persistable](../persistable.md)

**Data Type Compatibility:** Depends on distance kernel

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | k | 5 | int | The number of nearest neighbors to consider when making a prediction. |
| 2 | weighted | false | bool | Should we consider the distances of our nearest neighbors when making predictions? |
| 3 | kernel | Euclidean | Distance | The distance kernel used to compute the distance between sample points. |

## Example
```php
use Rubix\ML\Classifiers\KNearestNeighbors;
use Rubix\ML\Kernels\Distance\Manhattan;

$estimator = new KNearestNeighbors(3, false, new Manhattan());
```

## Additional Methods
This estimator does not have any additional methods.
