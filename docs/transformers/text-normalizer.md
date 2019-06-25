<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Transformers/TextNormalizer.php">Source</a></span>

# Text Normalizer
This transformer converts all text to lowercase and *optionally* removes extra whitespace.

**Interfaces:** [Transformer](api.md#transformer)

**Data Type Compatibility:** Categorical

### Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | trim | false | bool | Should we trim excess whitespace? |

### Additional Methods
This transformer does not have any additional methods.

### Example
```php
use Rubix\ML\Transformers\TextNormalizer;

$transformer = new TextNormalizer(true);
```
