<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Datasets/Extractors/NDJSON.php">[source]</a></span>

# NDJSON
[NDJSON](http://ndjson.org/) or *Newline Delimited* JSON files contain rows of data encoded in Javascript Object Notation (JSON). The rows can either be given as JSON arrays with integer keys or objects with string keys. One advantage NDJSON has over the [CSV](csv.md) format is that it retains data type information although it can have a slightly heavier footprint.

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