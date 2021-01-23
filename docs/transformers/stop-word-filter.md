<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Transformers/StopWordFilter.php">[source]</a></span>

# Stop Word Filter
Removes user-specified words from any categorical feature columns including blobs of text.

**Interfaces:** [Transformer](api.md#transformer)

**Data Type Compatibility:** Categorical

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | stopWords | | array | A list of stop words to filter out of each text feature. |

## Example
```php
use Rubix\ML\Transformers\StopWordFilter;

$transformer = new StopWordFilter(['i', 'me', 'my', ...]);
```

## Additional Methods
This transformer does not have any additional methods.
