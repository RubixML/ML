### Stop Word Filter
Removes user-specified words from any categorical feature column including blobs of text.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Transformers/StopWordFilter.php)

**Interfaces:** [Transformer](#transformers)

**Compatibility:** Categorical

**Parameters:**

| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | stop words | | array | A list of stop words to filter out of each text feature. |

**Additional Methods:**

This transformer does not have any additional methods.

```php
use Rubix\ML\Transformers\StopWordFilter;

$transformer = new StopWordFilter(['i', 'me', 'my', ...]);
```