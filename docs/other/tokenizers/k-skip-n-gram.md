<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Other/Tokenizers/KSkipNGram.php">[source]</a></span>

# K-Skip-N-Gram
K-skip-n-grams are a technique similar to n-grams, whereby n-grams are formed but in addition to allowing adjacent sequences of words, the next *k* words will be skipped forming n-grams of the new forward looking sequences. The tokenizer outputs tokens ranging from *min* to *max* number of words per token.

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | min | 2 | int | The minimum number of words in a single token. |
| 2 | max | 2 | int | The maximum number of words in a single token. |
| 3 | skip | 2 | int | The number of words to skip over to form new sequences. |

## Example
```php
use Rubix\ML\Other\Tokenizers\KSkipNGram;

$tokenizer = new KSkipNGram(2, 3, 2);
```
