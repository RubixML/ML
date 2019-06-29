<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/AnomalyDetectors/IsolationForest.php">Source</a></span>

# Isolation Forest
An ensemble anomaly detector comprised of Isolation Trees each trained on a different subset of the training set. Isolation Trees are a type of randomized decision tree that assigns an isolation score based on the depth in the tree a sample reaches upon search. The Isolation Forest works by averaging the isolation score of a sample across a user-specified number of trees.

**Interfaces:** [Estimator](../estimator.md), [Learner](../learner.md), [Ranking](api.md#ranking), [Persistable](../persistable.md)

**Data Type Compatibility:** Categorical, Continuous

### Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | estimators | 300 | int | The number of isolation trees to train in the ensemble. |
| 2 | contamination | 0.1 | float | The percentage of outliers that are assumed to be present in the training set. |
| 3 | ratio | 0.2 | float | The ratio of samples to train each estimator with. If *null* then subsample size is 256. |

### Additional Methods
This estimator does not have any additional methods.

### Example
```php
use Rubix\ML\AnomalyDetection\IsolationForest;

$estimator = new IsolationForest(100, 0.2, 0.01);

$estimator = new IsolationForest(100, null, 0.01); // Default sample size
```

### References
>- F. T. Liu et al. (2008). Isolation Forest.
>- F. T. Liu et al. (2011). Isolation-based Anomaly Detection.