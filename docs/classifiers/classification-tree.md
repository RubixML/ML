<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Classifiers/ClassificationTree.php">[source]</a></span>

# Classification Tree
A binary tree-based learner that greedily constructs a decision map for classification that minimizes the Gini impurity among the training labels within the leaf nodes. The height and *bushiness* of the tree can be determined by the user-defined `max height` and `max leaf size` hyper-parameters. Classification Trees also serve as the base learner of ensemble methods such as [Random Forest](random-forest.md) and [AdaBoost](adaboost.md).

**Interfaces:** [Estimator](../estimator.md), [Learner](../learner.md), [Probabilistic](../probabilistic.md), [Ranks Features](../ranks-features.md), [Persistable](../persistable.md)

**Data Type Compatibility:** Categorical, Continuous

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | maxHeight | PHP_INT_MAX | int | The maximum height of the tree. |
| 2 | maxLeafSize | 3 | int | The max number of samples that a leaf node can contain. |
| 3 | minPurityIncrease | 1e-7 | float | The minimum increase in purity necessary to continue splitting a subtree. |
| 4 | maxFeatures | Auto | int | The max number of feature columns to consider when determining a best split. |
| 5 | maxBins | Auto | int | The maximum number of bins to consider when determining a split with a continuous feature as the split point. |

## Example
```php
use Rubix\ML\Classifiers\ClassificationTree;

$estimator = new ClassificationTree(10, 5, 0.001, null, null);
```

## Additional Methods
Export a Graphviz "dot" encoding of the decision tree structure.
```php
public exportGraphviz() : Encoding
```

```php
use Rubix\ML\Helpers\Graphviz;
use Rubix\ML\Persisters\Filesystem;

$dot = $estimator->exportGraphviz();

Graphviz::dotToImage($dot)->saveTo(new Filesystem('tree.png'));
```

Return the number of levels in the tree.
```php
public height() : ?int
```

Return a factor that quantifies the skewness of the distribution of nodes in the tree.
```php
public balance() : ?int
```

## References:
[^1]: W. Y. Loh. (2011). Classification and Regression Trees.
[^2]: K. Alsabti. et al. (1998). CLOUDS: A Decision Tree Classifier for Large Datasets.
