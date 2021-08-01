<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Transformers/TextNormalizer.php">[source]</a></span>

# Text Normalizer
Converts all the characters in a blob of text to the same case.

**Interfaces:** [Transformer](api.md#transformer)

**Data Type Compatibility:** Categorical

!!! note
    This transformer does not handle multibyte strings. For multibyte support, see [MultibyteTextNormalizer](multibyte-text-normalizer.md).

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | uppercase | false | bool | Should the text be converted to uppercase? |

## Example
```php
use Rubix\ML\Transformers\TextNormalizer;

$transformer = new TextNormalizer(false);
```

## Additional Methods
This transformer does not have any additional methods.

