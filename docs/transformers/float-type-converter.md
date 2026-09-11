<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Transformers/FloatTypeConverter.php">[source]</a></span>

# Float Type Converter

Convert all integer and numeric string values to their equivalent floating point type. Useful for when continuous features are inadvertently stored as integers by either the PHP interpreter or JSON serialization, or as strings by the extraction from a source that only recognizes data as string types such as CSV. Both of these cases would otherwise cause the features to be inferred as categorical data.

!!! note
    The string representations of the PHP constants `NAN` and `INF` are the case-insensitive string literals 'NAN' and 'INF' respectively.

**Interfaces:** [Transformer](api.md#transformer)

**Data Type Compatibility:** Categorical, Continuous

## Parameters

This transformer does not have any parameters.

## Example

```php
use Rubix\ML\Transformers\FloatTypeConverter;

$transformer = new FloatTypeConverter();
```

## Additional Methods

This transformer does not have any additional methods.
