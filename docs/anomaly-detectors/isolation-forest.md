### Isolation Forest
An ensemble detector comprised of Isolation Trees each trained on a different subset of the training set. The Isolation Forest works by averaging the isolation score of a sample across a user-specified number of trees.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/AnomalyDetectors/IsolationForest.php)

**Interfaces:** [Estimator](#estimators), [Learner](#learner), [Ranking](#ranking), [Persistable](#persistable)

**Compatibility:** Categorical, Continuous

**Parameters:**

| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | estimators | 300 | int | The number of estimators to train in the ensemble. |
| 2 | contamination | 0.1 | float | The percentage of outliers that are assumed to be present in the training set. |
| 3 | ratio | 0.2 | float | The ratio of random samples to train each estimator with. |

**Additional Methods:**

This estimator does not have any additional methods.

**Example:**

```php
use Rubix\ML\AnomalyDetection\IsolationForest;

$estimator = new IsolationForest(300, 0.01, 0.2);
```

**References:**
>- F. T. Liu et al. (2008). Isolation Forest.
>- F. T. Liu et al. (2011). Isolation-based Anomaly Detection.