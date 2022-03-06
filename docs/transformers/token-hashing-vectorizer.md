<span style="float:right;"><a href="https://github.com/RubixML/Extras/blob/master/src/Transformers/TokenHashingVectorizer.php">[source]</a></span>

# Token Hashing Vectorizer
Token Hashing Vectorizer builds token count vectors on the fly by employing a *hashing trick*. It is a stateless transformer that uses a hashing algorithm to assign token occurrences to a bucket in a vector of user-specified dimensionality. The advantage of hashing over storing a fixed vocabulary is that there is no memory footprint however there is a chance that certain tokens will collide with other tokens especially in lower-dimensional vector spaces.

!!! note
    The default hashing function is CRC32 and is a good mix between speed and output space utilization. MurmurHash has even greater utilization but at the cost of some speed and it is only available on PHP 8.1 and above. FNV1 is comparable to CRC32 but with slightly more overhead.

**Interfaces:** [Transformer](api.md#transformer)

**Data Type Compatibility:** Categorical

## Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | dimensions | | int | The dimensionality of the vector space. |
| 2 | tokenizer | Word | Tokenizer | The tokenizer used to extract tokens from blobs of text. |
| 3 | hashFn | callable | 'crc32' | The hash function that accepts a string token and returns an integer. |

## Example
```php
use Rubix\ML\Transformers\TokenHashingVectorizer;
use Rubix\ML\Tokenizers\Word();

$transformer = new TokenHashingVectorizer(10000, new Word(), TokenHashingVectorizer::MURMUR3);
```

## Additional Constants
The CRC32 callback function.
```php
public const CRC32 callable(string):int
```

The MurmurHash3 callback function.
```php
public const MURMUR3 callable(string):int
```

The FNV1 callback function.
```php
public const FNV1 callable(string):int
```

## Additional Methods
The MurmurHash3 hashing function:
```php
public static murmur3(string $input) : int
```

!!! note
    MurmurHash3 is only available on PHP 8.1 or above.

The FNV1a 32-bit hashing function:
```php
public static fnv1a32(string $input) : int
```