<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Transformers/MultibyteTextNormalizer.php">[source]</a></span>

# Multibyte Text Normalizer
This transformer converts all [multibyte text](https://www.php.net/manual/en/intro.mbstring.php) to lowercase and removes extra whitespace.

Multibyte string contains multibyte characters sush as accented characters (é, è, à, ...), emojis (😀, 😉, ...) or characters of non roman alphabets (Chinese, Cyrillic, ...). 

**Interfaces:** [Transformer](api.md#transformer)

**Data Type Compatibility:** Categorical

> **Note:** ⚠️ We recommend you install the [Multibyte string extension](https://www.php.net/manual/en/book.mbstring.php) for best performance.

## Parameters
This transformer does not have any parameters.

## Additional Methods
This transformer does not have any additional methods.

## Example
```php
use Rubix\ML\Transformers\MultibyteTextNormalizer;

$transformer = new MultibyteTextNormalizer();
```
