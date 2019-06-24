<p><span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/AnomalyDetectors/LocalOutlierFactor.php">Source</a></span></p>

# Local Outlier Factor
Local Outlier Factor (LOF) measures the local deviation of density of a given sample with respect to its *k* nearest neighbors. As such, LOF only considers the local region (or *neighborhood*) of an unknown sample which enables it to detect anomalies within individual clusters of data.

**Interfaces:** [Estimator](#estimators), [Learner](#learner), [Online](#online), [Ranking](#ranking), [Persistable](#persistable)

**Data Type Compatibility:** Continuous

### Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | k | 20 | int | The k nearest neighbors that form a local region. |
| 2 | contamination | 0.1 | float | The percentage of outliers that are assumed to be present in the training set. |
| 3 | kernel | Euclidean | object | The distance kernel used to compute the distance between sample points. |

### Additional Methods
This estimator does not have any additional methods.

### Example
```php
use Rubix\ML\AnomalyDetection\LocalOutlierFactor;
use Rubix\ML\Kernels\Distance\Minkowski;

$estimator = new LocalOutlierFactor(20, 0.1, new Minkowski(3.5));
```

### References
>- M. M. Breunig et al. (2000). LOF: Identifying Density-Based Local Outliers.