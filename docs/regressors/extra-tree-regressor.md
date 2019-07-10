<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Regressors/ExtraTreeRegressor.php">Source</a></span>

# Extra Tree Regressor
An *Extremely Randomized* Regression Tree, these trees differ from standard [Regression Trees](regression-tree.md) in that they choose a split drawn from a random set determined by the max features parameter, rather than searching the entire column for the best split.

**Interfaces:** [Estimator](../estimator.md), [Learner](../learner.md), [Persistable](../persistable.md)

**Data Type Compatibility:** Categorical, Continuous

### Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | max depth | PHP_INT_MAX | int | The maximum depth of a branch in the tree. |
| 2 | max leaf size | 3 | int | The max number of samples that a leaf node can contain. |
| 3 | min purity increase | 0. | float | The minimum increase in purity necessary for a node *not* to be post pruned. |
| 4 | max features | Auto | int | The number of features to consider when determining a best split. |
| 5 | tolerance | 1e-4 | float | A small amount of impurity to tolerate when choosing a best split. |

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
use Rubix\ML\Classifiers\ExtraTreeRegressor;

$estimator = new ExtraTreeRegressor(30, 3, 0.05, 20, 1e-4);
```

### References
>- P. Geurts et al. (2005). Extremely Randomized Trees.