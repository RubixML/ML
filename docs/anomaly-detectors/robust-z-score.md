<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/AnomalyDetectors/RobustZScore.php">Source</a></span>

# Robust Z Score
A statistical anomaly detector that uses modified Z scores that are robust to preexisting outliers. The modified Z score takes the median and median absolute deviation (MAD) unlike the mean and standard deviation of a *standard* Z score - thus making the statistic more robust to training sets that may already contain outliers. Anomalies are flagged if their maximum feature-specific Z score exceeds some user-defined threshold parameter.

**Interfaces:** [Estimator](../estimator.md), [Learner](../learner.md), [Ranking](../ranking.md), [Persistable](../persistable.md)

**Data Type Compatibility:** Continuous

### Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | threshold | 3.5 | float | The minimum average Z score to be considered an anomaly. |

### Additional Methods
Return the median of each feature column in the training set:
```php
public medians() : ?array
```

Return the median absolute deviation (MAD) of each feature column in the training set:
```php
public mads() : ?array
```

### Example
```php
use Rubix\ML\AnomalyDetection\RobustZScore;

$estimator = new RobustZScore(3.0);
```

### References
>- P. J. Rousseeuw et al. (2017). Anomaly Detection by Robust Statistics.