<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Datasets/Extractors/NDJSON.php">[source]</a></span>

# NDJSON
[NDJSON](http://ndjson.org/) or *Newline Delimited* JSON files contain rows of data encoded in Javascript Object Notation (JSON) arrays or objects. The format is similar to CSV but has the advantage of being standardized and retaining data type information at the cost of having a slightly heavier footprint.

> **Note:** Empty rows will be ignored by the parser by default.

## Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | path |  | string | The path to the NDJSON file. |

## Additional Methods
This extractor does not have any additional methods.

## Example
```php
use Rubix\ML\Datasets\Extractors\NDJSON;

$extractor = new NDJSON('example.ndjson');
```