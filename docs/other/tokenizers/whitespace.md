### Whitespace
Tokens are delimited by a user-specified whitespace character.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Other/Tokenizers/Whitespace.php)

**Parameters:**

| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | delimiter | ' ' | string | The whitespace character that delimits each token. |

**Example:**

```php
use Rubix\ML\Extractors\Tokenizers\Whitespace;

$tokenizer = new Whitespace(',');
```