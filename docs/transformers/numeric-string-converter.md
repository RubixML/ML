<p><span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Transformers/NumericStringConverter.php">Source</a></span></p>

# Numeric String Converter
Convert all numeric strings into their integer and floating point countertypes. Useful for when extracting from a source that only recognizes data as string types.

**Interfaces:** [Transformer](#transformers)

**Data Type Compatibility:** Categorical

### Parameters
This transformer does not have any parameters.

### Additional Methods
This transformer does not have any additional methods.

### Example
```php
use Rubix\ML\Transformers\NumericStringConverter;

$transformer = new NumericStringConverter();
```