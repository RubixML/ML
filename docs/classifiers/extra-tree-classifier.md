<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Classifiers/ExtraTreeClassifier.php">[source]</a></span>

# Extra Tree Classifier
An *Extremely Randomized* Classification Tree that recursively chooses node splits with the least entropy among a set of *k* (given by max features) random split points. Extra Trees are useful in ensembles such as [Random Forest](random-forest.md) or [AdaBoost](adaboost.md) as the *weak* learner or they can be used on their own. The strength of Extra Trees as compared to standard decision trees are their computational efficiency and lower prediction variance.

**Interfaces:** [Estimator](../estimator.md), [Learner](../learner.md), [Probabilistic](../probabilistic.md), [Ranks Features](../ranks-features.md), [Persistable](../persistable.md)

**Data Type Compatibility:** Categorical, Continuous

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | maxHeight | PHP_INT_MAX | int | The maximum height of the tree. |
| 2 | maxLeafSize | 3 | int | The max number of samples that a leaf node can contain. |
| 3 | minPurityIncrease | 1e-7 | float | The minimum increase in purity necessary to continue splitting a subtree. |
| 4 | maxFeatures | Auto | int | The max number of feature columns to consider when determining a best split. |

## Example
```php
use Rubix\ML\Classifiers\ExtraTreeClassifier;

$estimator = new ExtraTreeClassifier(50, 3, 1e-7, 10);
```

## Additional Methods
Export a Graphviz "dot" encoding of the decision tree structure.
```php
public exportGraphviz() : Encoding
```

Return the number of levels in the tree.
```php
public height() : ?int
```

Return a factor that quantifies the skewness of the distribution of nodes in the tree.
```php
public balance() : ?int
```

## References
[^1]: P. Geurts et al. (2005). Extremely Randomized Trees.