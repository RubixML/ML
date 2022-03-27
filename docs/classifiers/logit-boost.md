<span style="float:right;"><a href="https://github.com/RubixML/Extras/blob/master/src/Classifiers/LogitBoost.php">[source]</a></span>

# Logit Boost
A stage-wise additive ensemble that uses regression trees to iteratively learn a Logistic Regression model for binary classification problems. Unlike standard [Logistic Regression](logistic-regression.md), Logit Boost has the ability to learn a smooth non-linear decision surface by training decision trees to follow the gradient of the cross entropy loss function. In addition, Logit Boost concentrates more effort on classifying samples that it is less certain about.

!!! note
    Logit Boost utilizes progress monitoring via an internal validation set for snapshotting and early stopping. If there are not enough training samples to build an internal validation set given the user-specified holdout ratio then training will proceed with progress monitoring disabled.

**Interfaces:** [Estimator](../estimator.md), [Learner](../learner.md), [Probabilistic](../probabilistic.md), [Verbose](../verbose.md), [Ranks Features](../ranks-features.md), [Persistable](../persistable.md)

**Data Type Compatibility:** Depends on base learners

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | booster | RegressionTree | Learner | The regressor used to fix up the error residuals of the base learner. |
| 2 | rate | 0.1 | float | The learning rate of the ensemble i.e. the *shrinkage* applied to each step. |
| 3 | ratio | 0.5 | float | The ratio of samples to subsample from the training set to train each booster. |
| 4 | epochs | 1000 | int | The maximum number of training epochs. i.e. the number of times to iterate before terminating. |
| 5 | minChange | 1e-4 | float | The minimum change in the training loss necessary to continue training. |
| 6 | window | 5 | int | The number of epochs without improvement in the validation score to wait before considering an early stop. |
| 7 | holdOut | 0.1 | float | The proportion of training samples to use for internal validation. Set to 0 to disable. |
| 8 | metric | F Beta | Metric | The metric used to score the generalization performance of the model during training. |

## Example
```php
use Rubix\ML\Classifiers\LogitBoost;
use Rubix\ML\Regressors\RegressionTree;
use Rubix\ML\CrossValidation\Metrics\FBeta;

$estimator = new LogitBoost(new RegressionTree(4), 0.1, 0.5, 1000, 1e-4, 5, 0.1, new FBeta());
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

Return the validation score for each epoch from the last training session:
```php
public scores() : float[]|null
```

Return the loss for each epoch from the last training session:
```php
public losses() : float[]|null
```

## References
[^1]: J. H. Friedman et al. (2000). Additive Logistic Regression: A Statistical View of Boosting.
[^2]: J. H. Friedman. (2001). Greedy Function Approximation: A Gradient Boosting Machine.
[^3]: J. H. Friedman. (1999). Stochastic Gradient Boosting.
[^4]: Y. Wei. et al. (2017). Early stopping for kernel boosting algorithms: A general analysis with localized complexities.
[^5]: G. Ke et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree.
