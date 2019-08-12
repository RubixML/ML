<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/AnomalyDetectors/RobustZScore.php">Source</a></span>

# Gaussian KDE
The Gaussian Kernel Density Estimator is able to spot outliers by computing a probability density function over the features assuming they are independent and normally (Gaussian) distributed. Assigning low probability density translates to a high anomaly score. The final anomaly score is given as the negative log likelihood of a sample being an outlier.

**Interfaces:** [Estimator](../estimator.md), [Learner](../learner.md), [Online](../online.md), [Ranking](../ranking.md), [Persistable](../persistable.md)

**Data Type Compatibility:** Continuous

### Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | threshold | 5.0 | float | The minimum negative log likelihood to be flagged as an anomaly. |
| 2 | contamination | 0.1 | float | The percentage of outliers that are assumed to be present in the training set. |

### Additional Methods
Return the column means computed from the training set:
```php
public means() : array
```

Return the column variances computed from the training set:
```php
public variances() : array
```

### Example
```php
use Rubix\ML\AnomalyDetection\GaussianKDE;

$estimator = new GaussianKDE(6.0, 0.1);
```

### References
>- B. Iglewicz et al. (1993). How to Detect and Handle Outliers.