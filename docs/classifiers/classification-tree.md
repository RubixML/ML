<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Classifiers/ClassificationTree.php">Source</a></span>

# Classification Tree
A binary search tree-based learner that greedily constructs a decision tree for classification that minimizes Gini impurity.

**Interfaces:** [Estimator](../estimator.md), [Learner](../learner.md), [Probabilistic](../probabilistic.md), [Persistable](../persistable.md)

**Data Type Compatibility:** Categorical, Continuous

### Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | max depth | PHP_INT_MAX | int | The maximum depth of a branch in the tree. |
| 2 | max leaf size | 3 | int | The max number of samples that a leaf node can contain. |
| 3 | max features | Auto | int | The max number of feature columns to consider when determining a best split. |
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
use Rubix\ML\Classifiers\ClassificationTree;

$estimator = new ClassificationTree(10, 7, 4, 0.01);
```

### References:
>- W. Y. Loh. (2011). Classification and Regression Trees.
>- K. Alsabti. et al. (1998). CLOUDS: A Decision Tree Classifier for Large Datasets.