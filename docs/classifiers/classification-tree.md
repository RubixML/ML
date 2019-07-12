<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Classifiers/ClassificationTree.php">Source</a></span>

# Classification Tree
A binary tree-based learner that minimizes gini impurity as a metric to greedily construct a decision tree used for classification.

**Interfaces:** [Estimator](../estimator.md), [Learner](../learner.md), [Probabilistic](../probabilistic.md), [Persistable](../persistable.md)

**Data Type Compatibility:** Categorical, Continuous

### Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | max depth | PHP_INT_MAX | int | The maximum depth of a branch in the tree. |
| 2 | max leaf size | 3 | int | The max number of samples that a leaf node can contain. |
| 3 | min purity increase | 0. | float | The minimum increase in purity necessary for a node *not* to be post pruned. |
| 4 | max features | Auto | int | The max number of feature columns to consider when determining a best split. |

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
use Rubix\ML\Classifiers\ClassificationTree;

// Import labeled dataset

$estimator = new ClassificationTree(10, 7, 0.1, 4);

$estimator->train($dataset);

var_dump($estimator->height());
var_dump($estimator->balance());

$estimator->showRules();
```

**Output**

```sh
int(4)
int(-1)

|--- Column_2 < 201.51545378637
|---|--- Column_0 < 206.79184370211
|---|---|--- Column_2 < -25.106356486194
|---|---|---|--- green
|---|---|--- Column_2 >= -25.106356486194
|---|---|---|--- green
|---|--- Column_0 >= 206.79184370211
|---|---|--- Column_2 < -45.599242677302
|---|---|---|--- red
|---|---|--- Column_2 >= -45.599242677302
|---|---|---|--- red
|--- Column_2 >= 201.51545378637
|---|--- Column_1 < -25.593268745045
|---|---|--- blue
|---|--- Column_1 >= -25.593268745045
|---|---|--- blue
```