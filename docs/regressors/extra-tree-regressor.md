<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Regressors/ExtraTreeRegressor.php">[source]</a></span>

# Extra Tree Regressor
An *Extremely Randomized* Regression Tree. These trees differ from standard [Regression Trees](regression-tree.md) in that they choose candidate splits at random, rather than searching the entire column for the best split. Extra Trees are faster to build and their predictions have higher variance than a regular decision tree.

**Interfaces:** [Estimator](../estimator.md), [Learner](../learner.md), [Ranks Features](../ranks-features.md), [Persistable](../persistable.md)

**Data Type Compatibility:** Categorical, Continuous

## Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | max depth | PHP_INT_MAX | int | The maximum depth of a branch in the tree. |
| 2 | max leaf size | 3 | int | The max number of samples that a leaf node can contain. |
| 3 | max features | Auto | int | The number of features to consider when determining a best split. |
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
use Rubix\ML\Regressors\ExtraTreeRegressor;

$estimator = new ExtraTreeRegressor(30, 3, 20, 0.05);
```

### References
>- P. Geurts et al. (2005). Extremely Randomized Trees.
