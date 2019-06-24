### Classification Tree
A binary tree-based classifier that minimizes gini impurity to greedily construct a decision tree for classification.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Classifiers/ClassificationTree.php)

**Interfaces:** [Estimator](#estimators), [Learner](#learner), [Probabilistic](#probabilistic), [Verbose](#verbose), [Persistable](#persistable)

**Compatibility:** Categorical, Continuous

**Parameters:**

| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | max depth | PHP_INT_MAX | int | The maximum depth of a branch. |
| 2 | max leaf size | 3 | int | The max number of samples that a leaf node can contain. |
| 3 | min purity increase | 0. | float | The minimum increase in purity necessary for a node *not* to be post pruned. |
| 4 | max features | Auto | int | The max number of features to consider when determining a best split. |
| 5 | tolerance | 1e-3 | float | A small amount of impurity to tolerate when choosing a best split. |

**Additional Methods:**

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

**Example:**

```php
use Rubix\ML\Classifiers\ClassificationTree;

$estimator = new ClassificationTree(30, 7, 0.1, 4, 1e-4);
```