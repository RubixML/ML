<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Transformers/MultibyteTextNormalizer.php">[source]</a></span>

# Multibyte Text Normalizer
This transformer converts the characters in all [multibyte strings](https://www.php.net/manual/en/intro.mbstring.php) to lowercase. Multibyte strings contain characters such as accents (é, è, à), emojis (😀, 😉) or characters of non roman alphabets such as Chinese and Cyrillic.

!!! note
    ⚠️ We recommend you install the [mbstring extension](https://www.php.net/manual/en/book.mbstring.php) for best performance.
 
**Interfaces:** [Transformer](api.md#transformer)

**Data Type Compatibility:** Categorical

## Parameters
This transformer does not have any parameters.

## Example
```php
use Rubix\ML\Transformers\MultibyteTextNormalizer;

$transformer = new MultibyteTextNormalizer();
```

## Additional Methods
This transformer does not have any additional methods.

