<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Regressors/RegressionTree.php">Source</a></span>

# Regression Tree
A Decision Tree learning algorithm (CART) that performs greedy splitting by minimizing the impurity (variance) of the labels at each decision node split.

**Interfaces:** [Estimator](../estimator.md), [Learner](../learner.md), [Persistable](../persistable.md)

**Data Type Compatibility:** Categorical, Continuous

### Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | max depth | PHP_INT_MAX | int | The maximum depth of a branch in the tree. |
| 2 | max leaf size | 3 | int | The maximum number of samples that a leaf node can contain. |
| 3 | min purity increase | 0. | float | The minimum increase in purity necessary for a node *not* to be post pruned. |
| 4 | max features | Auto | int | The maximum number of features to consider when determining a best split. |

### Additional Methods
Return the feature importances calculated during training indexed by feature column:
```php
public featureImportances() : array
```

Return the height of the tree:
```php
public height() : int
```

Return the balance of the tree:
```php
public balance() : int
```

Print out a human readable text representation of the decision tree:
```php
public showRules() : void
```

### Example
```php
use Rubix\ML\Regressors\RegressionTree;

$estimator = new RegressionTree(30, 2, 35., null);
```