<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Regressors/ExtraTreeRegressor.php">[source]</a></span>

# Extra Tree Regressor
*Extremely Randomized* Regression Trees differ from standard [Regression Trees](regression-tree.md) in that they choose candidate splits at random rather than searching the entire feature column for the best value to split on. Extra Trees are also faster to build and their predictions have higher variance than a regular decision tree regressor.

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
use Rubix\ML\Regressors\ExtraTreeRegressor;

$estimator = new ExtraTreeRegressor(30, 3, 20, 0.05);
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

## References
[^1]: P. Geurts et al. (2005). Extremely Randomized Trees.
