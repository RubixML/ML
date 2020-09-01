<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Extractors/NDJSON.php">[source]</a></span>

# NDJSON
[NDJSON](http://ndjson.org/) or *Newline Delimited* JSON files contain rows of data encoded in Javascript Object Notation (JSON) arrays or objects. The format is like a mix of JSON and CSV and has the advantage of retaining data type information and being read into memory incrementally.

> **Note:** Empty lines are ignored by the parser.

## Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | path |  | string | The path to the NDJSON file. |
| 2 | filesystem | `Storage::local()`  | `Filesystem` | The filesystem to read from. The default behaviour is to target the local filesystem.  |

## Example
```php
use Rubix\ML\Extractors\NDJSON;

$extractor = new NDJSON('example.ndjson');
```

## Additional Methods
This extractor does not have any additional methods.
