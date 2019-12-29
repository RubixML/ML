<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Extractors/ColumnPicker.php">[source]</a></span>

# Column Picker
An extractor that wraps another iterator and selects and rearranges the columns of the data table according to the user-specified keys. The key of a column may either be a string or a column number (integer) depending on the base iterator.

## Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | iterator | | Traversable | The base iterator. |
| 2 | keys | | array | The keys of the columns to iterate over. |

## Additional Methods
This extractor does not have any additional methods.

## Example
```php
use Rubix\ML\Extractors\ColumnPicker;
use Rubix\ML\Extractors\CSV;

$extractor = new ColumnPicker(new CSV('example.csv', true), [
    'attitude', 'texture', 'class', 'rating',
]);
```
