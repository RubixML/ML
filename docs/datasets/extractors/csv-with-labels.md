<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Datasets/Extractors/CSVWithLabels.php">[source]</a></span>

# CSV With Labels
A version of the CSV extractor where the last column of the data table is taken as the values for the label of a labeled dataset.

### Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | path |  | string | The path to the CSV file. |
| 2 | delimiter | ',' | string | The character that delineates a new column. |
| 3 | enclosure | '' | string | The character used to enclose the value of a column. |

### Example
```php
use Rubix\ML\Datasets\Extractors\CSVWithLabels;

$extractor = new CSVWithLabels('example.csv', ',', '');
```