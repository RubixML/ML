<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Classifiers/RandomForest.php">[source]</a></span>

# Random Forest
Random Forest (RF) is a classifier that trains an ensemble of Decision Trees ([Classification Trees](classification-tree.md) or [Extra Trees](extra-tree-classifier.md)) on random subsets (*bootstrap* set) of the training data. Predictions are based on the probability scores returned from each tree in the ensemble, averaged and weighted equally. In addition to reliable predictions, Random Forest also returns reliable feature importance scores making it suitable for feature selection.

!!! note
    The default base tree learner is a fully grown [Classification Tree](classification-tree.md).

**Interfaces:** [Estimator](../estimator.md), [Learner](../learner.md), [Probabilistic](../probabilistic.md), [Parallel](../parallel.md), [Ranks Features](../ranks-features.md), [Persistable](../persistable.md)

**Data Type Compatibility:** Categorical, Continuous

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | base | ClassificationTree | Learner | The base learner. |
| 2 | estimators | 100 | int | The number of learners to train in the ensemble. |
| 3 | ratio | 0.2 | float | The ratio of samples from the training set to randomly subsample to train each base learner. |
| 4 | balanced | false | bool | Should we sample the bootstrap set to compensate for imbalanced class labels? |

## Example
```php
use Rubix\ML\Classifiers\RandomForest;
use Rubix\ML\Classifiers\ClassificationTree;

$estimator = new RandomForest(new ClassificationTree(10), 300, 0.1, true);
```

## Additional Methods
This estimator does not have any additional methods.

## References
[^1]: L. Breiman. (2001). Random Forests.
[^2]: L. Breiman et al. (2005). Extremely Randomized Trees.