<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/AnomalyDetectors/RobustZScore.php">[source]</a></span>

# Robust Z-Score
A statistical anomaly detector that uses modified Z-Scores that are robust to preexisting outliers in the training set. The modified Z-Score is defined as the feature value minus the median over the median absolute deviation (MAD). Anomalies are flagged if their final weighted Z-Score exceeds a user-defined threshold.

!!! note
    An alpha value of 1 means the estimator only considers the maximum absolute Z-Score, whereas a setting of 0 indicates that only the average Z-Score factors into the final score.

**Interfaces:** [Estimator](../estimator.md), [Learner](../learner.md), [Scoring](../scoring.md), [Persistable](../persistable.md)

**Data Type Compatibility:** Continuous

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | threshold | 3.5 | float | The minimum Z-Score to be flagged as an anomaly. |
| 2 | alpha | 0.5 | float | The weight of the maximum per-sample Z-Score in the overall anomaly score. |

## Example
```php
use Rubix\ML\AnomalyDetectors\RobustZScore;

$estimator = new RobustZScore(3.0, 0.3);
```

## Additional Methods
Return the median of each feature column in the training set:
```php
public medians() : float[]|null
```

Return the median absolute deviation (MAD) of each feature column in the training set:
```php
public mads() : float[]|null
```

## References
[^1]: B. Iglewicz et al. (1993). How to Detect and Handle Outliers.