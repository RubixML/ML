<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Tokenizers/Whitespace.php">[source]</a></span>

# Whitespace
Tokens are delimited by a user-specified whitespace character.

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | delimiter | ' ' | string | The whitespace character that delimits each token. |

## Example
```php
use Rubix\ML\Tokenizers\Whitespace;

$tokenizer = new Whitespace(',');
```
