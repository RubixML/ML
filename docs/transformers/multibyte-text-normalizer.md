<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Transformers/MultibyteTextNormalizer.php">[source]</a></span>

# Multibyte Text Normalizer
This transformer converts the characters in all [multibyte strings](https://www.php.net/manual/en/intro.mbstring.php) to the same case. Multibyte strings contain characters such as accents (é, è, à), emojis (😀, 😉) or characters of non roman alphabets such as Chinese and Cyrillic.

!!! note
    ⚠️ We recommend you install the [mbstring extension](https://www.php.net/manual/en/book.mbstring.php) for best performance.
 
**Interfaces:** [Transformer](api.md#transformer)

**Data Type Compatibility:** Categorical

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | uppercase | false | bool | Should the text be converted to uppercase? |

## Example
```php
use Rubix\ML\Transformers\MultibyteTextNormalizer;

$transformer = new MultibyteTextNormalizer(false);
```

## Additional Methods
This transformer does not have any additional methods.
