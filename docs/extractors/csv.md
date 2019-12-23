<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Datasets/Extractors/CSV.php">[source]</a></span>

# CSV
A non-standard plain-text format that use newlines to delineate rows and a user-specified delimiter (usually a comma) to separate the values of each column in a data table. Comma-Separated Values (CSV) format is a common format but suffers from the disadvantage of not being able to retain type information - thus, all data is imported as categorical data (strings) by default.

### Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | path |  | string | The path to the CSV file. |
| 2 | delimiter | ',' | string | The character that delineates a new column. |
| 4 | enclosure | '' | string | The character used to enclose the value of a column. |

### Additional Methods
Does the CSV document have a header as the first row?:
```php
public setHeader(bool $header = true) : self
```

### Example
```php
use Rubix\ML\Datasets\Extractors\CSV;

$extractor = new CSV('example.csv', ',', '"');

$extractor->setHeader(true);
```