<span style="float:right;"><a href="https://github.com/RubixML/Extras/blob/master/src/Transformers/TokenHashingVectorizer.php">[source]</a></span>

# Token Hashing Vectorizer
Token Hashing Vectorizer builds token count vectors on the fly by employing a *hashing trick*. It is a stateless transformer that uses the CRC32 (Cyclic Redundancy Check) hashing algorithm to assign token occurrences to a bucket in a vector of user-specified dimensionality. The advantage of hashing over storing a fixed vocabulary is that there is no memory footprint however there is a chance that certain tokens will collide with other tokens especially in lower-dimensional vector spaces.

**Interfaces:** [Transformer](api.md#transformer)

**Data Type Compatibility:** Categorical

## Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | dimensions | | int | The dimensionality of the vector space. |
| 2 | tokenizer | Word | Tokenizer | The tokenizer used to extract tokens from blobs of text. |

## Example
```php
use Rubix\ML\Transformers\TokenHashingVectorizer;
use Rubix\ML\Tokenizers\NGram;

$transformer = new TokenHashingVectorizer(10000, new NGram(1, 2));
```

## Additional Methods
This transformer does not have any additional methods.
