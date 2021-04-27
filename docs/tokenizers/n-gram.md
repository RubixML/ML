<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Tokenizers/NGram.php">[source]</a></span>

# N-gram
N-grams are sequences of n-words of a given string. The N-gram tokenizer outputs tokens of contiguous words ranging from *min* to *max* number of words per token.

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | min | 2 | int | The minimum number of contiguous words to a token. |
| 2 | max | 2 | int | The maximum number of contiguous words to a token. |

## Example
```php
use Rubix\ML\Tokenizers\NGram;

$tokenizer = new NGram(1, 3);
```
