<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Extractors/CSV.php">[source]</a></span>

# CSV
A plain-text format that use newlines to delineate rows and a user-specified delimiter (usually a comma) to separate the values of each column in a data table. Comma-Separated Values (CSV) format is a common format but suffers from not being able to retain type information - thus, all data is imported as categorical data (strings) by default.

!!! note
    This implementation of CSV is based on the definition in [RFC 4180](https://tools.ietf.org/html/rfc4180).

**Interfaces:** [Extractor](api.md), [Writable](api.md)

## Parameters
| # | Name      | Default | Type | Description |
|---|---|---|---|---|
| 1 | path | | string | The path to the CSV file. |
| 2 | header | false | bool | Does the CSV document have a header as the first row? |
| 3 | delimiter | ',' | string | The character that delineates the values of the columns of the data table. |
| 4 | enclosure | '"' | string | The character used to enclose a cell that contains a delimiter in the body. |
| 5 | escape | '\\' | string | The character used as an escape character (one character only). |

## Example
```php
use Rubix\ML\Extractors\CSV;

$extractor = new CSV('example.csv', true, ',', '"','\\');
```

## Additional Methods
Return the column titles of the data table.
```php
public header() : array
```

## References
[^1]: T. Shafranovich. (2005). Common Format and MIME Type for Comma-Separated Values (CSV) Files.