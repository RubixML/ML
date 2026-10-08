<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Tokenizers/KSkipNGram.php">[source]</a></span>

# K-Skip-N-Gram

K-skip-n-grams are a technique similar to n-grams, whereby n-grams are formed but in addition to allowing adjacent sequences of words, the next *k* words will be skipped forming n-grams of the new forward looking sequences. The tokenizer outputs tokens ranging from *min* to *max* number of words per token.

## Parameters

| # | Name | Default | Type | Description |
| --- | --- | --- | --- | --- |
| 1 | min | 2 | int | The minimum number of words in a single token. |
| 2 | max | 2 | int | The maximum number of words in a single token. |
| 3 | skip | 2 | int | The number of words to skip over to form new sequences. |

## Example

```php
use Rubix\ML\Tokenizers\KSkipNGram;

$tokenizer = new KSkipNGram(2, 3, 2);
```

## How The Skip Works

The *skip* parameter is the widest stride considered between the words of a single token, and every stride from `0` up to *skip* is emitted. A stride of `0` produces the plain (adjacent) n-gram, so this tokenizer is a superset of the [N-Gram](n-gram.md) tokenizer.

For a stride of *k*, the words of a token are taken from positions `i`, `i + k + 1`, `i + 2(k + 1)`, and so on. Consider `a b c d e` with `min: 3`, `max: 3`, `skip: 1`:

| Stride | Token | Words |
| --- | --- | --- |
| 0 | `a b c` | `a`, `b`, `c` |
| 0 | `b c d` | `b`, `c`, `d` |
| 0 | `c d e` | `c`, `d`, `e` |
| 1 | `a c e` | `a`, `c`, `e` |
| 1 | `b c d` | `b`, `c`, `d` |
| 1 | `c d e` | `c`, `d`, `e` |

Raising *skip* to `2` widens the stride further, adding `a d g` from `a b c d e f g` in addition to the `0` and `1` stride tokens. Strides that would reach past the end of a sentence are simply not emitted, so short sentences yield fewer tokens rather than partial ones.
