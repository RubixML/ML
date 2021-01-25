<span style="float:right;"><a href="https://github.com/RubixML/Extras/blob/master/src/Transformers/KBestFeatureSelector.php">[source]</a></span>

# K Best Feature Selector
A supervised feature selector that picks the top K ranked features returned by a learner implementing the [RanksFeatures](../ranks-features.md) interface.

!!! note
    The default feature ranking base learner is a fully-grown decision tree.

**Interfaces:** [Transformer](api.md#transformer), [Stateful](api.md#stateful), [Persistable](../persistable.md)

**Data Type Compatibility:** Depends on the base learner

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | k | | int | The maximum number of features to select from the dataset. |
| 2 | scorer | Auto | RanksFeatures | The base feature scorer. |

## Additional Methods
Return the final importances of the selected feature columns:
``` php
public importances() : ?array
```

## Example
```php
use Rubix\ML\Transformers\KBestFeatureSelector;
use Rubix\ML\Regressors\GradientBoost;

$transformer = new KBestFeatureSelector(10, new GradientBoost());
```
