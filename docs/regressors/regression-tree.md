<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Regressors/RegressionTree.php">Source</a></span>

# Regression Tree
A Decision Tree learning algorithm (CART) that performs greedy splitting by minimizing the variance of of the labels at each split.

**Interfaces:** [Estimator](../estimator.md), [Learner](../learner.md), [Persistable](../persistable.md)

**Data Type Compatibility:** Categorical, Continuous

### Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | max depth | PHP_INT_MAX | int | The maximum depth of a branch in the tree. |
| 2 | max leaf size | 3 | int | The maximum number of samples that a leaf node can contain. |
| 3 | max features | Auto | int | The maximum number of features to consider when determining a best split. |
| 4 | min purity increase | 1e-7 | float | The minimum increase in purity necessary for a node *not* to be post pruned. |

### Additional Methods
Return the normalized feature importances i.e. the proportion that each feature contributes to the overall model, indexed by feature column:
```php
public featureImportances() : array
```

Display a human readable text representation of the decision tree:
```php
public printrules() : void
```

Return the height of the tree:
```php
public height() : int
```

Return the balance factor of the tree:
```php
public balance() : int
```

### Example
```php
use Rubix\ML\Regressors\RegressionTree;

$estimator = new RegressionTree(20, 2, null, 1e-3);
```

### References:
>- W. Y. Loh. (2011). Classification and Regression Trees.
>- K. Alsabti. et al. (1998). CLOUDS: A Decision Tree Classifier for Large Datasets.