<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Extractors/ARFF.php">[source]</a></span>

# ARFF

The Attribute-Relation File Format (ARFF) is an ASCII text format that is native to the Weka machine learning workbench. Along with being widely used in academic research, ARFF files retain the data type of each column via the attribute declarations in the header of the file.

!!! note
    Missing values, denoted by a question mark (`?`), are imported as `NAN` for numeric and real attributes and as the categorical placeholder string for integer, date, and categorical attributes. The placeholder defaults to `?`. Integer attributes are imported as PHP integers.

!!! note
    Single-quoted strings may contain a literal single quote as a backslash-escaped apostrophe (`\'`). Multi-line quoted strings and trailing `%` comments after such values are still recognized correctly.

**Interfaces:** [Extractor](api.md)

## Parameters

| # | Name | Default | Type | Description |
| --- | --- | --- | --- | --- |
| 1 | path | | string | The path to the ARFF file on disk. |
| 2 | categoricalPlaceholder | '?' | string\|int | The string or integer to substitute in place of missing date and categorical values. |

## Example

```php
use Rubix\ML\Extractors\ARFF;

$extractor = new ARFF('dataset.arff', 'unknown');
```

## Additional Methods

Return the column titles of the data table.

```php
public header() : array
```

## References

[^1]: I. H. Witten et al. (1999). WEKA - Data Mining with the Java Algorithms for Machine Learning Workbench.