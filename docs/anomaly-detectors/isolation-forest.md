<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/AnomalyDetectors/IsolationForest.php">[source]</a></span>

# Isolation Forest
An ensemble of Isolation Trees that each specialize on a unique subset of the training set. Isolation Trees are a type of randomized decision tree that assign anomaly scores based on the depth a sample reaches when traversing the tree. Based on the premise that anomalies are isolated into their own nodes sooner, samples that receive high anomaly scores achieve the shallowest depth during traversal.

**Interfaces:** [Estimator](../estimator.md), [Learner](../learner.md), [Scoring](../scoring.md), [Persistable](../persistable.md)

**Data Type Compatibility:** Categorical, Continuous

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | estimators | 100 | int | The number of isolation trees to train in the ensemble. |
| 2 | ratio | null | float | The ratio of samples to train each estimator with. If null, the subsample size will be set to 256. |
| 3 | contamination | null | float | The proportion of outliers that are assumed to be present in the training set. If null, the threshold anomaly score will be set to 0.5. |

## Example
```php
use Rubix\ML\AnomalyDetectors\IsolationForest;

$estimator = new IsolationForest(100, 0.2, 0.05);
```

## Additional Methods
This estimator does not have any additional methods.

## References
[^1]: F. T. Liu et al. (2008). Isolation Forest.
[^2]: F. T. Liu et al. (2011). Isolation-based Anomaly Detection.
[^3]: M. Garchery et al. (2018). On the influence of categorical features in ranking anomalies using mixed data.
