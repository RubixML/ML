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
| 3 | maxFeatures | Auto | int | The max number of feature columns to consider when determining a best split. |
| 4 | minPurityIncrease | 1e-7 | float | The minimum increase in purity necessary for a node *not* to be post pruned during tree growth. |

## Example
```php
use Rubix\ML\Regressors\RegressionTree;

$estimator = new RegressionTree(20, 2, null, 1e-3);
```

## Additional Methods
Return a human-readable text representation of the decision tree ruleset:
```php
public rules(?array $header = null) : string
```

Return the height of the tree i.e. the number of layers:
```php
public height() : int
```

Return the balance factor of the tree:
```php
public balance() : int
```

## References:
[^1]: W. Y. Loh. (2011). Classification and Regression Trees.
[^2]: K. Alsabti. et al. (1998). CLOUDS: A Decision Tree Classifier for Large Datasets.