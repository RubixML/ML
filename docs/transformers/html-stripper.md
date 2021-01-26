<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Transformers/HTMLStripper.php">[source]</a></span>

# HTML Stripper
Removes any HTML or PHP tags from the text of a feature column.

!!! note
    Since the HTML is not actually validated during transformation, broken tags may result in unexpectedly removing non-HTML text.

**Interfaces:** [Transformer](api.md#transformer)

**Data Type Compatibility:** Categorical

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | allowedTags | | array | A list of html tags that should not be stripped ex. ['p', 'br']. |

## Example
```php
use Rubix\ML\Transformers\HTMLStripper;

$transformer = new HTMLStripper();
```

## Additional Methods
This transformer does not have any additional methods.
