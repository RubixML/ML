<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Extractors/ColumnFilter.php">[source]</a></span>

# Column Filter

**Interfaces:** [Extractor](api.md)

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | iterator | | Traversable | The base iterator. |
| 2 | keys | | array | The string and/or integer keys of the columns to filter from the table |

## Example
```php
use Rubix\ML\Extractors\ColumnFilter;
use Rubix\ML\Extractors\CSV;

$extractor = new ColumnFilter(new CSV('example.csv', true), [
    'texture', 'class',
]);
```

## Additional Methods
This extractor does not have any additional methods.
