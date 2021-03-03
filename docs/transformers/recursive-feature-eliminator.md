<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Transformers/RecursiveFeatureEliminator.php">[source]</a></span>

# Recursive Feature Eliminator
Recursive Feature Eliminator (RFE) is a supervised feature selector that uses the importance scores returned by a learner implementing the [RanksFeatures](../ranks-features.md) interface to recursively drop feature columns with the lowest importance until a terminating condition is met.

!!! note
    The default feature ranking base learner is a fully-grown decision tree.

**Interfaces:** [Transformer](api.md#transformer), [Stateful](api.md#stateful), [Verbose](../verbose.md), [Persistable](../persistable.md)

**Data Type Compatibility:** Depends on the base learner

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | minFeatures | | int | The minimum number of features to select from the dataset. |
| 2 | maxDroppedFeatures | 3 | int | The maximum number of features to drop from the dataset per iteration. |
| 3 | maxDroppedImportance | 0.2 | float | The maximum importance to drop from the dataset per iteration. |
| 4 | scorer | Auto | RanksFeatures | The base feature scorer. |

## Additional Methods
Return the final importances of the selected feature columns:
``` php
public importances() : ?array
```

## Example
```php
use Rubix\ML\Transformers\RecursiveFeatureEliminator;
use Rubix\ML\Classifiers\RandomForest;

$transformer = new RecursiveFeatureEliminator(30, 2, 0.05 new RandomForest());
```

## References
[^1]: I. Guyon et al. (2002). Gene Selection for Cancer Classification using Support Vector Machines.
