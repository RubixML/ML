<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Transformers/BooleanConverter.php">[source]</a></span>

# Boolean Converter
This transformer is used to convert boolean values to a compatible continuous or categorical datatype. Strings should be
used when the boolean should be treated as a categorical value. Ints or floats when the boolean should be treated as a
continuous value.

**Interfaces:** [Transformer](api.md#transformer)

**Data Type Compatibility:** Categorical, Continuous

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | trueValue | 'true' | string, int, float | The value to convert `true` to. |
| 2 | falseValue | 'false' | string, int, float | The value to convert `false` to. |

## Example
```php
use Rubix\ML\Transformers\BooleanConverter;

$transformer = new BooleanConverter('true', 'false);

$transformer = new BooleanConverter('tall', 'not tall');

$transformer = new BooleanConverter(1, 0);
```

## Additional Methods
This transformer does not have any additional methods.
