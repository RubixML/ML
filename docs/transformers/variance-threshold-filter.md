<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Transformers/VarianceThresholdFilter.php">[source]</a></span>

# Variance Threshold Filter
A type of feature selector that selects the top *k* features with the greatest variance.

**Interfaces:** [Transformer](api.md#transformer), [Stateful](api.md#stateful)

**Data Type Compatibility:** Continuous

## Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | maxFeatures | | int | The maximum number of features to select from the dataset. |

## Additional Methods
Return the offsets of the columns that were selected during fitting:
```php
public selected() : array
```

Return the variances of the selected feature columns:
```php
public variances() : ?array
```

## Example
```php
use Rubix\ML\Transformers\VarianceThresholdFilter;

$transformer = new VarianceThresholdFilter(50);
```