<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/AnomalyDetectors/KDLOF.php">Source</a></span>

# Local Outlier Factor
Local Outlier Factor (LOF) measures the local deviation of density of a given sample with respect to its *k* nearest neighbors. As such, LOF only considers the local region (or *neighborhood*) of an unknown sample which enables it to detect anomalies within individual clusters of data.

**Interfaces:** [Estimator](../estimator.md), [Learner](../learner.md), [Ranking](api.md#ranking), [Persistable](../persistable.md)

**Data Type Compatibility:** Continuous

### Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | k | 20 | int | The k nearest neighbors that form a local region. |
| 2 | contamination | null | float | The percentage of outliers that are assumed to be present in the training set. |
| 3 | kernel | Euclidean | object | The distance kernel used to compute the distance between sample points. |
| 4 | max leaf size | 30 | int | The max number of samples in a leaf node (*neighborhood*). |

### Additional Methods
Return the base spatial tree instance:
```php
public tree() : Spatial
```

### Example
```php
use Rubix\ML\AnomalyDetection\KDLOF;
use Rubix\ML\Graph\Trees\KDTree;
use Rubix\ML\Kernels\Distance\Euclidean;

$estimator = new KDLOF(20, 0.1, new KDTree(30, new Euclidean));
```

### References
>- M. M. Breunig et al. (2000). LOF: Identifying Density-Based Local Outliers.