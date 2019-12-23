<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Transformers/RegexFilter.php">[source]</a></span>

# Regex Filter
Filters the text columns of a dataset by matching a list of regular expressions.

**Interfaces:** [Transformer](api.md#transformer)

**Data Type Compatibility:** Categorical

## Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | patterns | | array | A list of regular expression patterns used to filter the text columns of the dataset. |

## Additional Methods
This transformer does not have any additional methods.

## Example
```php
use Rubix\ML\Transformers\RegexFilter;

$transformer = new RegexFilter([
    RegexFilter::PATTERNS['url'],
    RegexFilter::PATTERNS['mention'],
    '/(?<andrew>.+)/',
]);
```