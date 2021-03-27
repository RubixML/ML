<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Classifiers/KDNeighbors.php">[source]</a></span>

# K-d Neighbors
A fast K Nearest Neighbors algorithm that uses a binary search tree (BST) to divide the training set into *neighborhoods* that contain samples that are close together spatially. K-d Neighbors then does a binary search to locate the nearest neighborhood of an unknown sample and prunes all neighborhoods whose bounding box is further than the *k*'th nearest neighbor found so far. The main advantage of K-d Neighbors over brute force [KNN](k-nearest-neighbors.md) is that it is much more efficient, however it cannot be partially trained.

**Interfaces:** [Estimator](../estimator.md), [Learner](../learner.md), [Probabilistic](../probabilistic.md), [Persistable](../persistable.md)

**Data Type Compatibility:** Depends on distance kernel

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | k | 5 | int | The number of nearest neighbors to consider when making a prediction. |
| 2 | weighted | false | bool | Should we consider the distances of our nearest neighbors when making predictions? |
| 3 | tree | KDTree | Spatial | The spatial tree used to run nearest neighbor searches. |

## Example
```php
use Rubix\ML\Classifiers\KDNeighbors;
use Rubix\ML\Graph\Trees\BallTree;
use Rubix\ML\Kernels\Distance\Minkowski;

$estimator = new KDNeighbors(10, false, new BallTree(40, new Minkowski()));
```

## Additional Methods
Return the base spatial tree instance:
```php
public tree() : Spatial
```
