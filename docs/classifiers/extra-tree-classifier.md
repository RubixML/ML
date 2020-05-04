<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Classifiers/ExtraTreeClassifier.php">[source]</a></span>

# Extra Tree Classifier
An *Extremely Randomized* Classification Tree that recursively chooses node splits with the least entropy among a set of *k* (given by max features) random split points. Extra Trees are useful in ensembles such as [Random Forest](random-forest.md) or [AdaBoost](adaboost.md) as the *weak* learner or they can be used on their own. The strength of Extra Trees as compared to standard decision trees are their computational efficiency and lower variance.

**Interfaces:** [Estimator](../estimator.md), [Learner](../learner.md), [Probabilistic](../probabilistic.md), [Ranks Features](../ranks-features.md), [Persistable](../persistable.md)

**Data Type Compatibility:** Categorical, Continuous

## Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | max depth | PHP_INT_MAX | int | The maximum depth of a branch in the tree. |
| 2 | max leaf size | 3 | int | The max number of samples that a leaf node can contain. |
| 3 | max features | Auto | int | The max number of features columns to consider when determining a best split. |
| 4 | min purity increase | 1e-7 | float | The minimum increase in purity necessary for a node *not* to be post pruned. |

## Additional Methods
Return a human readable text representation of the decision tree ruleset:
```php
public rules() : string
```

Return the height of the tree:
```php
public height() : int
```

Return the balance factor of the tree:
```php
public balance() : int
```

## Example
```php
use Rubix\ML\Classifiers\ExtraTreeClassifier;

$estimator = new ExtraTreeClassifier(50, 3, 4, 1e-7);
```

### References
>- P. Geurts et al. (2005). Extremely Randomized Trees.