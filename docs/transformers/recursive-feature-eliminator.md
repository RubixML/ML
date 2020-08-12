<span style="float:right;"><a href="https://github.com/RubixML/Extras/blob/master/src/Transformers/RecursiveFeatureEliminator.php">[source]</a></span>

# Recursive Feature Eliminator
Recursive Feature Eliminator or *RFE* is a supervised feature selector that uses the importance scores returned by a learner implementing the RanksFeatures interface to recursively drop feature columns with the lowest importance until max features is reached.

**Interfaces:** [Transformer](api.md#transformer), [Stateful](api.md#stateful)

**Data Type Compatibility:** Depends on the base learner

## Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | max features | | int | The maximum number of features to select. |
| 2 | epochs | 1 | int | The maximum number of iterations to recurse upon the dataset. |
| 3 | base | Auto | RanksFeatures | The base feature ranking learner instance. |

## Additional Methods
Return the final importances scores of the selected feature columns:
``` php
public importances() : ?array
```

## Example
```php
use Rubix\ML\Transformers\RecursiveFeatureEliminator;
use Rubix\ML\Regressors\RegressionTree;

$transformer = new RecursiveFeatureEliminator(10, 2, new RegressionTree());
```

### References
>- I. Guyon et al. (2002). Gene Selection for Cancer Classification using Support Vector Machines.
