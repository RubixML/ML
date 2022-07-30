<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Regressors/RegressionTree.php">[source]</a></span>

# Regression Tree
A decision tree based on the CART (*Classification and Regression Tree*) learning algorithm that performs greedy splitting by minimizing the variance of the labels at each node split. Regression Trees can be used on their own or as the booster in algorithms such as [Gradient Boost](gradient-boost.md).

**Interfaces:** [Estimator](../estimator.md), [Learner](../learner.md), [Ranks Features](../ranks-features.md), [Persistable](../persistable.md)

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
use Rubix\ML\Regressors\RegressionTree;

$estimator = new RegressionTree(20, 2, 1e-3, 10, null);
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