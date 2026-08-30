<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Classifiers/AdaBoost.php">[source]</a></span>

# AdaBoost
Short for *Adaptive Boosting*, this ensemble classifier can improve the performance of an otherwise *weak* classifier by focusing more attention on samples that are harder to classify. It builds an additive model where, at each stage, a new learner is trained and given an influence score inversely proportional to the loss it incurs at that epoch. In addition, the learner features progress monitoring which stops training when it can no longer improve the validation score.

!!! note
    The default base learner is a [Classification Tree](classification-tree.md) with a max height of 1 i.e a *Decision Stump*.

!!! note
    If there are not enough training samples to build an internal validation set with the user-specified holdout ratio then progress monitoring will be disabled.

**Interfaces:** [Estimator](../estimator.md), [Learner](../learner.md), [Probabilistic](../probabilistic.md), [Verbose](../verbose.md), [Persistable](../persistable.md)

**Data Type Compatibility:** Depends on base learner

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | base | ClassificationTree | Learner | The base *weak* classifier to be boosted. |
| 2 | rate | 1.0 | float | The learning rate of the ensemble i.e. the *shrinkage* applied to each step. |
| 3 | ratio | 0.8 | float | The ratio of samples to subsample from the training set to train each *weak* learner. |
| 4 | epochs | 100 | int | The maximum number of training epochs. i.e. the number of times to iterate before terminating. |
| 5 | minChange | 1e-4 | float | The minimum change in the training loss necessary to continue training. |
| 6 | evalInterval | 3 | int | The number of epochs to train before evaluating the model using the holdout set. |
| 7 | window | 5 | int | The number of epochs without improvement in the validation score to wait before considering an early stop. |
| 8 | holdOut | 0.1 | float | The proportion of training samples to use for internal validation. Set to 0 to disable. |
| 9 | metric | FBeta | Metric | The validation metric used to score the generalization performance of the model during training. |

## Example
```php
use Rubix\ML\Classifiers\AdaBoost;
use Rubix\ML\Classifiers\ExtraTreeClassifier;
use Rubix\ML\CrossValidation\Metrics\MCC;

$estimator = new AdaBoost(new ExtraTreeClassifier(3), 0.1, 0.5, 200, 1e-3, 3, 5, 0.1, new MCC());
```

## Additional Methods
Return an iterable progress table with the steps from the last training session:
```php
public steps() : iterable
```

```php
use Rubix\ML\Extractors\CSV;

$extractor = new CSV('progress.csv', true);

$extractor->export($estimator->steps());
```

Return the loss for each epoch from the last training session:
```php
public losses() : float[]|null
```

Return the validation score for each epoch from the last training session:
```php
public scores() : float[]|null
```

## References
[^1]: Y. Freund et al. (1996). A Decision-theoretic Generalization of On-line Learning and an Application to Boosting.
[^2]: J. Zhu et al. (2006). Multi-class AdaBoost.
