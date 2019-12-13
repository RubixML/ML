<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Datasets/Extractors/NDJSONWithLabels.php">[source]</a></span>

# NDJSON With Labels
Newline Delimited JSON with Labels uses the last column of an NDJSON data table as the value of the dataset's labels.

> **Note:** Empty rows will be ignored by the parser by default.

### Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | path |  | string | The path to the NDJSON file. |

### Example
```php
use Rubix\ML\Datasets\Extractors\NDJSONWithLabels;

$extractor = new NDJSONWithLabels('example.ndjson');
```