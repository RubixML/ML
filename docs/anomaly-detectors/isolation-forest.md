<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/AnomalyDetectors/IsolationForest.php">[source]</a></span>

# Isolation Forest
An ensemble learner comprised of Isolation Trees that are each trained on a unique subset of the training set. Isolation Trees are a type of randomized decision tree that assign anomaly scores based on the depth a sample reaches when traversing the tree from root to leaf node. Anomalies are isolated into their own nodes earliest during tree traversal and therefore receive the highest *isolation* scores. The Isolation Forest works by averaging the anomaly scores for an unknown sample across a user-specified number of trees.

**Interfaces:** [Estimator](../estimator.md), [Learner](../learner.md), [Ranking](../ranking.md), [Persistable](../persistable.md)

**Data Type Compatibility:** Categorical, Continuous

## Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | estimators | 100 | int | The number of isolation trees to train in the ensemble. |
| 2 | ratio | | float | The ratio of samples to train each estimator with. If `null` then the subsample size is 256. |
| 3 | contamination | null | float | The proportion of outliers that are assumed to be present in the training set. |

## Additional Methods
This estimator does not have any additional methods.

## Example
```php
use Rubix\ML\AnomalyDetectors\IsolationForest;

$estimator = new IsolationForest(100, 0.2, 0.03);

$estimator = new IsolationForest(100); // Default sample size and threshold
```

### References
>- F. T. Liu et al. (2008). Isolation Forest.
>- F. T. Liu et al. (2011). Isolation-based Anomaly Detection.
>- M. Garchery et al. (2018). On the influence of categorical features in ranking anomalies using mixed data.