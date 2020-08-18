<span style="float:right;"><a href="https://github.com/RubixML/Extras/blob/master/src/Transformers/RecursiveFeatureEliminator.php">[source]</a></span>

# Recursive Feature Eliminator
Recursive Feature Eliminator or *RFE* is a supervised feature selector that uses the importance scores returned by a learner implementing the [RanksFeatures](../ranks-features.md) interface to recursively drop feature columns with the lowest importance until the minimum number of features has been reached.

> **Note:** The default feature ranking base learner is a fully grown decision tree.

**Interfaces:** [Transformer](api.md#transformer), [Stateful](api.md#stateful), [Verbose](../verbose.md)

**Data Type Compatibility:** Depends on the base learner

## Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | min features | | int | The minimum number of features to select. |
| 2 | max drop features | 3 | int | The maximum number of features to drop from the dataset per iteration. |
| 3 | max drop importance | 0.2 | float | The maximum importance to drop from the dataset per iteration. |
| 4 | base | Auto | RanksFeatures | The base feature ranking learner instance. |

## Additional Methods
Return the final importances of the selected feature columns:
``` php
public importances() : ?array
```

## Example
```php
use Rubix\ML\Transformers\RecursiveFeatureEliminator;
use Rubix\ML\Regressors\Ridge;

$transformer = new RecursiveFeatureEliminator(30, 2, 0.05 new Ridge());
```

### References
>- I. Guyon et al. (2002). Gene Selection for Cancer Classification using Support Vector Machines.
