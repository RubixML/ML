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
| 3 | minPurityIncrease | 1e-7 | float | The minimum increase in purity necessary to continue splitting a subtree. |
| 4 | maxFeatures | Auto | int | The max number of feature columns to consider when determining a best split. |

## Example
```php
use Rubix\ML\Regressors\ExtraTreeRegressor;

$estimator = new ExtraTreeRegressor(30, 5, 0.05, null);
```

## Additional Methods
Export a Graphviz "dot" encoding of the decision tree structure.
```php
public exportGraphviz() : Encoding
```

Return the number of levels in the tree.
```php
public height() : ?int
```

Return a factor that quantifies the skewness of the distribution of nodes in the tree.
```php
public balance() : ?int
```

## References
[^1]: P. Geurts et al. (2005). Extremely Randomized Trees.
