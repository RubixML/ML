<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Classifiers/ClassificationTree.php">[source]</a></span>

# Classification Tree
A binary tree-based learner that greedily constructs a decision map for classification that minimizes the Gini impurity among the training labels within the leaf nodes. The height and *bushiness* of the tree can be determined by the user-defined `max depth` and `max leaf size` hyper-parameters. Classification Trees also serve as the base learner of ensemble methods such as [Random Forest](random-forest.md) and [AdaBoost](adaboost.md).

**Interfaces:** [Estimator](../estimator.md), [Learner](../learner.md), [Probabilistic](../probabilistic.md), [Ranks Features](../ranks-features.md), [Persistable](../persistable.md)

**Data Type Compatibility:** Categorical, Continuous

## Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | max depth | PHP_INT_MAX | int | The maximum depth of a branch in the tree. |
| 2 | max leaf size | 3 | int | The max number of samples that a leaf node can contain. |
| 3 | max features | Auto | int | The max number of feature columns to consider when determining a best split. |
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
use Rubix\ML\Classifiers\ClassificationTree;

$estimator = new ClassificationTree(10, 7, 4, 0.01);
```

### References:
>- W. Y. Loh. (2011). Classification and Regression Trees.
>- K. Alsabti. et al. (1998). CLOUDS: A Decision Tree Classifier for Large Datasets.