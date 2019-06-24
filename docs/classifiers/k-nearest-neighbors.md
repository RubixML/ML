<p><span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Classifiers/KNearestNeighbors.php">Source</a></span></p>

# K Nearest Neighbors
A distance-based algorithm that locates the K nearest neighbors from the training set and uses a weighted vote to classify the unknown sample.

> **Note**: K Nearest Neighbors is considered a *lazy* learner because it does the majority of its computation at inference. For a fast tree-based version, see [KD Neighbors](#k-d-neighbors).

**Interfaces:** [Estimator](#estimators), [Learner](#learner), [Online](#online), [Probabilistic](#probabilistic), [Persistable](#persistable)

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
use Rubix\ML\Classifiers\KNearestNeighbors;
use Rubix\ML\Kernels\Distance\Manhattan;

$estimator = new KNearestNeighbors(3, new Manhattan(), true);
```