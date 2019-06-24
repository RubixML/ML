<p><span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/AnomalyDetectors/RobustZScore.php">Source</a></span></p>

# Robust Z Score
A quick *global* anomaly detector that uses a modified Z score which is robust to outliers to detect anomalies within a dataset. The modified Z score consists of taking the median and median absolute deviation (MAD) instead of the mean and standard deviation (*standard* Z score) thus making the statistic more robust to training sets that may already contain outliers. Outliers can be flagged in one of two ways. First, their average Z score can be above the user-defined tolerance level or an individual feature's score could be above the threshold (*hard* limit).

**Interfaces:** [Estimator](#estimators), [Learner](#learner), [Persistable](#persistable)

**Data Type Compatibility:** Continuous

### Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | tolerance | 3.0 | float | The average z score to tolerate before a sample is considered an outlier. |
| 2 | threshold | 3.5 | float | The threshold z score of a individual feature to consider the entire sample an outlier. |

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

$estimator = new RobustZScore(1.5, 3.0);
```

### References
>- P. J. Rousseeuw et al. (2017). Anomaly Detection by Robust Statistics.