<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Other/Tokenizers/SkipGram.php">[source]</a></span>

# Skip-Gram
Skip-grams are a technique similar to n-grams, whereby n-grams are formed but in addition to allowing adjacent sequences of words, the next *k* words will be *skipped* forming n-grams of the new forward looking sequences.

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | n | 2 | int | The number of contiguous words to a single token. |
| 2 | skip | 2 | int | The number of words to skip over to form new n-gram sequences. |

## Example
```php
use Rubix\ML\Extractors\Tokenizers\SkipGram;

$tokenizer = new SkipGram(2, 2);
```