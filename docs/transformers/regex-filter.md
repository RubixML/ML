<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Transformers/RegexFilter.php">[source]</a></span>

# Regex Filter
Filters the text features of a dataset by matching and removing patterns from a list of regular expressions.

!!! note
    Patterns are filtered in the same sequence as they are given in the constructor.

**Interfaces:** [Transformer](api.md#transformer)

**Data Type Compatibility:** Categorical

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | patterns | | array | A list of regular expression patterns used to filter the text columns of the dataset. |

## Example
```php
use Rubix\ML\Transformers\RegexFilter;

$transformer = new RegexFilter([
    RegexFilter::URL,
    RegexFilter::MENTION,
    '/(?<me>.+)/',
    RegexFilter::EXTRA_CHARACTERS,
]);
```

## Predefined Regex Patterns
| Class Constant | Description |
|---|---|
| EMAIL | A pattern to match any email address. |
| URL | An alias for the default (Gruber 1) URL matching pattern. |
| GRUBER_1 | The original Gruber URL matching pattern. |
| GRUBER_2 | The improved Gruber URL matching pattern. |
| EXTRA_CHARACTERS | Matches consecutively repeated non word or number characters such as punctuation and special characters. |
| EXTRA_WORDS | Matches consecutively repeated words. |
| EXTRA_WHITESPACE | Matches consecutively repeated whitespace characters. |
| EMOJIS | A pattern to match unicode emojis. |
| MENTION | A pattern that matches Twitter-style mentions (@example). |
| HASHTAG | Matches Twitter-style hashtags (#example). |

## Additional Methods
This transformer does not have any additional methods.

## References:
[^1]: J. Gruber. (2009). A Liberal, Accurate Regex Pattern for Matching URLs.
[^2]: J. Gruber. (2010). An Improved Liberal, Accurate Regex Pattern for Matching URLs.
