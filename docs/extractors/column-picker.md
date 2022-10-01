<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Extractors/ColumnPicker.php">[source]</a></span>

# Column Picker
An extractor that wraps another iterator and selects and reorders the columns of the data table according to the keys specified by the user. The key of a column may either be a string or a column number (integer) depending on the way the columns are indexed in the base iterator.

**Interfaces:** [Extractor](api.md)

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | iterator | | Traversable | The base iterator. |
| 2 | keys | | array | The string and/or integer keys of the columns to pick and reorder from the table |

## Example
```php
use Rubix\ML\Extractors\ColumnPicker;
use Rubix\ML\Extractors\CSV;

$extractor = new ColumnPicker(new CSV('example.csv', true), [
    'attitude', 'texture', 'class', 'rating',
]);
```

## Additional Methods
This extractor does not have any additional methods.
