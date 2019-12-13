<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Datasets/Extractors/NDJSON.php">[source]</a></span>

# NDJSON
Newline Delimited JSON (NDJSON) files contain rows of Javascript Object Notation (JSON) encoded data. The rows can either be JSON arrays with integer keys or objects with string keys. One advantage NDJSON has over the CSV format is that it retains data type information.

> **Note:** Empty rows will be ignored by the parser by default.

### Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | path |  | string | The path to the NDJSON file. |

### Example
```php
use Rubix\ML\Datasets\Extractors\NDJSON;

$extractor = new NDJSON('example.ndjson');
```