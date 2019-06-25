<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Transformers/VarianceThresholdFilter.php">Source</a></span>

# Variance Threshold Filter
A type of feature selector that selects feature columns that have a greater variance than the user-specified threshold.

**Interfaces:** [Transformer](api.md#transformer), [Stateful](api.md#stateful)

**Data Type Compatibility:** Continuous

### Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | threshold | 0. | float | Feature columns with a variance greater than this threshold will be selected. |

### Additional Methods
Return the columns that were selected during fitting:
```php
public selected() : array
```

### Example
```php
use Rubix\ML\Transformers\VarianceThresholdFilter;

$transformer = new VarianceThresholdFilter(50);
```