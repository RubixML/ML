<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Transformers/VarianceThresholdFilter.php">[source]</a></span>

# Variance Threshold Filter
A type of feature selector that selects the top *k* features with the greatest variance.

**Interfaces:** [Transformer](api.md#transformer), [Stateful](api.md#stateful)

**Data Type Compatibility:** Continuous

## Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | minFeatures | | int | The minimum number of features to select from the dataset. |

## Example
```php
use Rubix\ML\Transformers\VarianceThresholdFilter;

$transformer = new VarianceThresholdFilter(50);
```

## Additional Methods
Return the variances of the dropped feature columns:
```php
public variances() : ?array
```
