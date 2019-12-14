<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Datasets/Extractors/NDJSONArray.php">[source]</a></span>

# NDJSON Array
[NDJSON](http://ndjson.org/) or *Newline Delimited* JSON files contain rows of data encoded in Javascript Object Notation (JSON) arrays. The format is similar to [CSV](csv.md) but has the advantage of retaining data type information at the cost of having a slightly heavier footprint.

> **Note:** Empty rows will be ignored by the parser by default.

### Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | path |  | string | The path to the NDJSON file. |

### Example
```php
use Rubix\ML\Datasets\Extractors\NDJSONArray;

$extractor = new NDJSONArray('example.ndjson');
```