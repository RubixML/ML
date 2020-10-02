<span style="float:right;"><a href="https://github.com/RubixML/Extras/blob/master/src/Transformers/KBestSelector.php">[source]</a></span>

# K Best Selector
A supervised feature selector that picks the top K ranked features returned by a learner implementing the [RanksFeatures](../ranks-features.md) interface.

> **Note:** The default feature ranking base learner is a fully-grown decision tree.

**Interfaces:** [Transformer](api.md#transformer), [Stateful](api.md#stateful)

**Data Type Compatibility:** Depends on the base learner

## Parameters
| # | Param | Default | Type | Description |
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
use Rubix\ML\Transformers\KBestSelector;
use Rubix\ML\Classifiers\GradientBoost;

$transformer = new KBestSelector(10, new GradientBoost());
```
