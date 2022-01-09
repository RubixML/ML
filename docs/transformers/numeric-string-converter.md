<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Transformers/NumericStringConverter.php">[source]</a></span>

# Numeric String Converter
Convert all numeric strings to their equivalent integer and floating point types. Useful for when extracting from a source that only recognizes data as string types such as CSV.

!!! note
    The string representations of the PHP constants `NAN` and `INF` are the string literals 'NAN' and 'INF' respectively. 

**Interfaces:** [Transformer](api.md#transformer), [Reversible](api.md#reversible)

**Data Type Compatibility:** Categorical

## Parameters
This transformer does not have any parameters.

## Example
```php
use Rubix\ML\Transformers\NumericStringConverter;

$transformer = new NumericStringConverter();
```

## Additional Methods
This transformer does not have any additional methods.
