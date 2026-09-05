<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Transformers/BooleanConverter.php">[source]</a></span>

# Boolean Converter

This transformer is used to convert truthy or falsy values to a another continuous or categorical datatype.

**Interfaces:** [Transformer](api.md#transformer)

**Data Type Compatibility:** Categorical, Continuous

## Parameters

| # | Name | Default | Type | Description |
| --- | --- | --- | --- | --- |
| 1 | trueValue | 1 | string, int, or float | The value to convert truthy to. |
| 2 | falseValue | 0 | string, int, or float | The value to convert falsy to. |

## Example

```php
use Rubix\ML\Transformers\BooleanConverter;

$transformer = new BooleanConverter(1, 0);

$transformer = new BooleanConverter('true', 'false');

$transformer = new BooleanConverter('tall', 'not tall');

$transformer = new BooleanConverter(5.0, -5.0);
```

## Additional Methods

This transformer does not have any additional methods.
