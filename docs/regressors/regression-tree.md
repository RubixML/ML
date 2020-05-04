<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Regressors/RegressionTree.php">[source]</a></span>

# Regression Tree
A decision tree based on the CART (*Classification and Regression Tree*) learning algorithm that performs greedy splitting by minimizing the variance of the labels at each node split.

**Interfaces:** [Estimator](../estimator.md), [Learner](../learner.md), [Ranks Features](../ranks-features.md), [Persistable](../persistable.md)

**Data Type Compatibility:** Categorical, Continuous

## Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | max depth | PHP_INT_MAX | int | The maximum depth of a branch in the tree. |
| 2 | max leaf size | 3 | int | The maximum number of samples that a leaf node can contain. |
| 3 | max features | Auto | int | The maximum number of features to consider when determining a best split. |
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
use Rubix\ML\Regressors\RegressionTree;

$estimator = new RegressionTree(20, 2, null, 1e-3);
```

### References:
>- W. Y. Loh. (2011). Classification and Regression Trees.
>- K. Alsabti. et al. (1998). CLOUDS: A Decision Tree Classifier for Large Datasets.