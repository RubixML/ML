### Text Normalizer
This transformer converts all text to lowercase and *optionally* removes extra whitespace.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Transformers/TextNormalizer.php)

**Interfaces:** [Transformer](#transformers)

**Compatibility:** Categorical

**Parameters:**

| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | trim | false | bool | Should we trim excess whitespace? |

**Additional Methods:**

This transformer does not have any additional methods.

```php
use Rubix\ML\Transformers\TextNormalizer;

$transformer = new TextNormalizer(true);
```
