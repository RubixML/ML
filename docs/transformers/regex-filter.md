<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Transformers/RegexFilter.php">[source]</a></span>

# Regex Filter
Filters the text columns of a dataset by matching a list of regular expressions.

**Interfaces:** [Transformer](api.md#transformer)

**Data Type Compatibility:** Categorical

## Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | patterns | | array | A list of regular expression patterns used to filter the text columns of the dataset. |

## Class Constants
| Name | Description |
|---|---|---|
| URL | An alias for the default (Gruber 1) URL matching pattern. |
| GRUBER_1 | The faster original Gruber URL matching pattern. |
| GRUBER_2 | The more universal improved Gruber URL matching pattern. |
| EMAIL | A pattern to match any email address. |
| MENTION | A pattern that matches Twitter-style mentions (@example). |
| HASHTAG | Matches Twitter-style hashtags (#example). |

## Additional Methods
This transformer does not have any additional methods.

## Example
```php
use Rubix\ML\Transformers\RegexFilter;

$transformer = new RegexFilter([
    RegexFilter::URL,
    RegexFilter::MENTION,
    '/(?<me>.+)/',
]);
```

### References:
>- J. Gruber. (2009). A Liberal, Accurate Regex Pattern for Matching URLs.
>- J. Gruber. (2010). An Improved Liberal, Accurate Regex Pattern for Matching URLs.