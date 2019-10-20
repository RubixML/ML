<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Transformers/NumericStringConverter.php">Source</a></span>

# Numeric String Converter
Convert all numeric strings into their integer and floating point countertypes. Useful for when extracting from a source that only recognizes data as string types such as CSV.

**Interfaces:** [Transformer](api.md#transformer)

**Data Type Compatibility:** Categorical

### Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | placeholder | 'NaN' | string | The placeholder string for NaN values. |

### Additional Methods
This transformer does not have any additional methods.

### Example
```php
use Rubix\ML\Transformers\NumericStringConverter;

$transformer = new NumericStringConverter('NaN');
```