<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Extractors/JSON.php">[source]</a></span>

# JSON
Javascript Object Notation is a standardized lightweight plain-text representation that is widely used. JSON has the advantage of retaining type information, however since the entire JSON blob is read on load, it is not cursorable like CSV or NDJSON.

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | path |  | string | The path to the JSON file. |

## Example
```php
use Rubix\ML\Extractors\JSON;

$extractor = new JSON('example.json');
```

## Additional Methods
This extractor does not have any additional methods.

## References
[^1]: T. Bray. (2014). The JavaScript Object Notation (JSON) Data Interchange Format.