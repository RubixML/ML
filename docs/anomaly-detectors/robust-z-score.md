<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/AnomalyDetectors/RobustZScore.php">[source]</a></span>

# Robust Z Score
A statistical anomaly detector that uses modified Z scores that are robust to preexisting outliers. The modified Z score uses the median and median absolute deviation (MAD) unlike the mean and standard deviation of a *standard* Z score which are sensitive to outliers. Anomalies are flagged if their final weighted Z score exceeds a user-defined threshold.

> **Note:** An alpha value of 1 means the estimator only considers the maximum absolute z score whereas a setting of 0 indicates that only the average z score factors into the final score.

**Interfaces:** [Estimator](../estimator.md), [Learner](../learner.md), [Ranking](../ranking.md), [Persistable](../persistable.md)

**Data Type Compatibility:** Continuous

### Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | threshold | 3.5 | float | The minimum Z score to be flagged as an anomaly. |
| 2 | alpha | 0.5 | float | The weight of the maximum per-sample Z score in the overall anomaly score. |

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
use Rubix\ML\AnomalyDetectors\RobustZScore;

$estimator = new RobustZScore(3.0, 0.3);
```

### References
>- B. Iglewicz et al. (1993). How to Detect and Handle Outliers.