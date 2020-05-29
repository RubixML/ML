<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Transformers/TextNormalizer.php">[source]</a></span>

# Text Normalizer
This transformer converts all text to lowercase and removes extra whitespace.

**Interfaces:** [Transformer](api.md#transformer)

**Data Type Compatibility:** Categorical

> **Note:** ⚠️ This transformer can't handle multibyte text properly. For multibyte text, use [MultibyteTextNormalizer](multibyte-text-normalizer.md).

## Parameters
This transformer does not have any parameters.

## Additional Methods
This transformer does not have any additional methods.

## Example
```php
use Rubix\ML\Transformers\TextNormalizer;

$transformer = new TextNormalizer();
```
