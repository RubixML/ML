<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Regressors/ExtraTreeRegressor.php">Source</a></span>

# Extra Tree Regressor
An *Extremely Randomized* Regression Tree. These trees differ from standard [Regression Trees](regression-tree.md) in that they choose a split drawn completely at random, rather than searching the entire column for the best split. Due to random splitting, Extra Trees are faster to build and their predictions have higher variance than a regular decision tree.

**Interfaces:** [Estimator](../estimator.md), [Learner](../learner.md), [Persistable](../persistable.md)

**Data Type Compatibility:** Categorical, Continuous

### Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | max depth | PHP_INT_MAX | int | The maximum depth of a branch in the tree. |
| 2 | max leaf size | 3 | int | The max number of samples that a leaf node can contain. |
| 3 | max features | Auto | int | The number of features to consider when determining a best split. |
| 4 | min purity increase | 1e-7 | float | The minimum increase in purity necessary for a node *not* to be post pruned. |

### Additional Methods
Return the normalized feature importances i.e. the proportion that each feature contributes to the overall model, indexed by feature column:
```php
public featureImportances() : array
```

Return a human readable text representation of the decision tree rules:
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

### Example
```php
use Rubix\ML\Classifiers\ExtraTreeRegressor;

$estimator = new ExtraTreeRegressor(30, 3, 20, 0.05);
```

### References
>- P. Geurts et al. (2005). Extremely Randomized Trees.