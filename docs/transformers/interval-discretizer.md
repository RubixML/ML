<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Transformers/IntervalDiscretizer.php">[source]</a></span>

# Interval Discretizer
Assigns each continuous feature to a discrete category using equi-width histograms. Useful for converting continuous data to categorical.

**Interfaces:** [Transformer](api.md#transformer), [Stateful](api.md#stateful)

**Data Type Compatibility:** Continuous

## Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | bins | 5 | int | The number of bins (discrete categories) per continuous feature column. |

## Additional Methods
Return the possible categories of each feature column:
```php
public categories() : array
```

Return the intervals of each continuous feature column calculated during fitting:
```php
public intervals() : array
```

## Example
```php
use Rubix\ML\Transformers\IntervalDiscretizer;

$transformer = new IntervalDiscretizer(10);
```