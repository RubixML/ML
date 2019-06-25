<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/AnomalyDetectors/KDLOF.php">Source</a></span>

# K-d LOF
A k-d tree accelerated version of [Local Outlier Factor](local-outlier-factor.md) which benefits from neighborhood pruning during nearest neighbors search. The tradeoff between K-d LOF and the brute force method is that while K-d LOF is faster, it cannot be partially trained.

**Interfaces:** [Estimator](../estimator.md), [Learner](../learner.md), [Ranking](api.md#ranking), [Persistable](../persistable.md)

**Data Type Compatibility:** Continuous

### Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | k | 20 | int | The k nearest neighbors that form a local region. |
| 2 | contamination | 0.1 | float | The percentage of outliers that are assumed to be present in the training set. |
| 3 | kernel | Euclidean | object | The distance kernel used to compute the distance between sample points. |
| 4 | max leaf size | 30 | int | The max number of samples in a leaf node (*neighborhood*). |

### Additional Methods
Return the base k-d tree instance:
```php
public tree() : KDTree
```

### Example
```php
use Rubix\ML\AnomalyDetection\KDLOF;
use Rubix\ML\Kernels\Distance\Euclidean;

$estimator = new KDLOF(20, 0.1, new Euclidean(), 30);
```

### References
>- M. M. Breunig et al. (2000). LOF: Identifying Density-Based Local Outliers.