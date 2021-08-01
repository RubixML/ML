<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Transformers/IntervalDiscretizer.php">[source]</a></span>

# Interval Discretizer
Assigns continuous features to ordered categories using variable width per-feature histograms with a fixed user-specified number of bins.

**Interfaces:** [Transformer](api.md#transformer), [Stateful](api.md#stateful), [Persistable](../persistable.md)

**Data Type Compatibility:** Continuous

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | bins | 5 | int | The number of bins per histogram. |
| 2 | equiWidth | false | bool | Should the bins be equal width? |

## Example
```php
use Rubix\ML\Transformers\IntervalDiscretizer;

$transformer = new IntervalDiscretizer(8, false);
```

## Additional Methods
Return the bin intervals of the fitted data:
```php
public intervals() : array
```
