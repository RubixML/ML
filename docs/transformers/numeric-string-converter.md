<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Transformers/NumericStringConverter.php">[source]</a></span>

# Numeric String Converter
Convert all numeric strings to their equivalent integer and floating point types. Useful for when extracting from a source that only recognizes data as string types such as CSV.

**Note:** NaN strings (i.e. `'NaN'`) are converted to their floating point equivalent.

**Interfaces:** [Transformer](api.md#transformer)

**Data Type Compatibility:** Categorical

## Parameters
This transformer does not have any parameters.

## Additional Methods
This transformer does not have any additional methods.

## Example
```php
use Rubix\ML\Transformers\NumericStringConverter;

$transformer = new NumericStringConverter();
```