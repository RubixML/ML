<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Transformers/MinMaxNormalizer.php">[source]</a></span>

# Min Max Normalizer
The *Min Max* Normalizer scales the input features to a value between a user-specified range (*default* 0 to 1).

**Interfaces:** [Transformer](api.md#transformer), [Stateful](api.md#stateful), [Elastic](api.md#elastic), [Reversible](api.md#reversible), [Persistable](../persistable.md)

**Data Type Compatibility:** Continuous

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | min | 0.0 | float | The minimum value of the transformed features. |
| 2 | max | 1.0 | float | The maximum value of the transformed features. |

## Example
```php
use Rubix\ML\Transformers\MinMaxNormalizer;

$transformer = new MinMaxNormalizer(-5.0, 5.0);
```

## Additional Methods
Return the minimum values for each fitted feature column:
```php
public minimums() : ?array
```

Return the maximum values for each fitted feature column:
```php
public maximums() : ?array
```
