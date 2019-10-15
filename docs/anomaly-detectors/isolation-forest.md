<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/AnomalyDetectors/IsolationForest.php">Source</a></span>

# Isolation Forest
An ensemble anomaly detector comprised of Isolation Trees (*ITrees*) trained on a unique subset of the training set. Isolation Trees are a type of randomized decision tree that assign *isolation* scores based on the depth a sample reaches in the tree. Outliers are said to be isolated earliest in the growing process and therefore receive higher isolation scores. The Isolation Forest works by averaging the isolation scores of a sample across a user-specified number of trees.

**Interfaces:** [Estimator](../estimator.md), [Learner](../learner.md), [Ranking](../ranking.md), [Persistable](../persistable.md)

**Data Type Compatibility:** Categorical, Continuous

### Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | estimators | 100 | int | The number of isolation trees to train in the ensemble. |
| 2 | ratio | Auto | float | The ratio of samples to train each estimator with. If *null* then subsample size is 256. |
| 3 | contamination | Auto | float | The proportion of outliers that are presumed to be present in the training set. |

### Additional Methods
This estimator does not have any additional methods.

### Example
```php
use Rubix\ML\AnomalyDetection\IsolationForest;

$estimator = new IsolationForest(100, 0.2, 0.03);

$estimator = new IsolationForest(100); // Default sample size and threshold
```

### References
>- F. T. Liu et al. (2008). Isolation Forest.
>- F. T. Liu et al. (2011). Isolation-based Anomaly Detection.
>- M. Garchery et al. (2018). On the influence of categorical features in ranking anomalies using mixed data.