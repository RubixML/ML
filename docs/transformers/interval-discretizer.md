<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Transformers/IntervalDiscretizer.php">[source]</a></span>

# Interval Discretizer
Converts each continuous feature to a category using equi-width histograms with a user-specified number of bins for discretization.

**Interfaces:** [Transformer](api.md#transformer), [Stateful](api.md#stateful), [Persistable](../persistable.md)

**Data Type Compatibility:** Continuous

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | bins | 5 | int | The number of bins (discrete categories) per continuous feature column. |

## Example
```php
use Rubix\ML\Transformers\IntervalDiscretizer;

$transformer = new IntervalDiscretizer(10);
```

## Additional Methods
Return the list of possible category values for each discretized feature column:
```php
public categories() : array
```

Return the intervals for each continuous feature column calculated during fitting:
```php
public intervals() : array
```
