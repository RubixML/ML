<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Transformers/WordOrderRandomizer.php">[source]</a></span>

# Word Randomizer
Splits the given text based on a separator and then shuffles it.

**Interfaces:** [Transformer](api.md#transformer)

**Data Type Compatibility:** Categorical

## Parameters
| # | Name      | Default | Type   | Description                                                     |
|---|-----------|---------|--------|-----------------------------------------------------------------|
| 1 | separator | ' '     | string | Should the transformer split the string based on ' ' character? |

## Example
```php
use Rubix\ML\Transformers\WordOrderRandomizer;

$transformer = new WordOrderRandomizer();
```

## Additional Methods
This transformer does not have any additional methods.

