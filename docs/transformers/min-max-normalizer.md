<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Transformers/MinMaxNormalizer.php">Source</a></span>

# Min Max Normalizer
The *Min Max* Normalizer scales the input features to a value between a user-specified range (*default* 0 to 1).

**Interfaces:** [Transformer](api.md#transformer), [Stateful](api.md#stateful), [Elastic](api.md#elastic)

**Data Type Compatibility:** Continuous

### Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | min | 0. | float | The minimum value of the transformed features. |
| 2 | max | 1. | float | The maximum value of the transformed features. |

### Additional Methods
Return the minimum values for each fitted feature column:
```php
public minimums() : ?array
```

Return the maximum values for each fitted feature column:
```php
public maximums() : ?array
```

### Example
```php
use Rubix\ML\Transformers\MinMaxNormalizer;

$transformer = new MinMaxNormalizer(-5., 5.);
```