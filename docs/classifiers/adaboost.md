<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Classifiers/AdaBoost.php">Source</a></span>

# AdaBoost
Short for *Adaptive Boosting*, this ensemble classifier can improve the performance of an otherwise *weak* classifier by focusing more attention on samples that are harder to classify. It builds an additive model where, at each stage, a new learner is instantiated and trained.

> **Note:** The default base classifier is a Classification Tree with a max depth of 1 i.e a *Decision Stump*.

**Interfaces:** [Estimator](../estimator.md), [Learner](../learner.md), [Probabilistic](../probabilistic.md), [Verbose](../verbose.md), [Persistable](../persistable.md)

**Data Type Compatibility:** Depends on base learner

### Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | base | ClassificationTree | object | The base *weak* classifier to be boosted. |
| 2 | estimators | 100 | int | The maximum number of estimators to train in the ensemble. |
| 3 | rate | 1.0 | float | The learning rate i.e step size. |
| 4 | ratio | 0.8 | float | The ratio of samples to subsample from the training set for each *weak* learner. |
| 5 | min change | 1e-4 | float | The minimum change in the training loss necessary to continue training. |

### Additional Methods
Return the calculated weight values of the last trained dataset:
```php
public weights() : array
```

Return the influence scores for each boosted classifier:
```php
public influences() : array
```

Return the training error at each epoch:
```php
public steps() : array
```

### Example
```php
use Rubix\ML\Classifiers\AdaBoost;
use Rubix\ML\Classifiers\ExtraTreeClassifier;

$estimator = new AdaBoost(new ExtraTreeClassifier(3), 100, 0.1, 0.5, 1e-3);
```

### References
 >- Y. Freund et al. (1996). A Decision-theoretic Generalization of On-line Learning and an Application to Boosting.
 >- J. Zhu et al. (2006). Multi-class AdaBoost.