<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Transformers/WordCountVectorizer.php">[source]</a></span>

# Word Count Vectorizer
The Word Count Vectorizer builds a vocabulary from the training samples and transforms text blobs into fixed length sparse feature vectors. Each feature column represents a word or *token* from the vocabulary and the value denotes the number of times that word appears in a given document.

**Interfaces:** [Transformer](api.md#transformer), [Stateful](api.md#stateful)

**Data Type Compatibility:** Categorical

## Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | max vocabulary | PHP_INT_MAX | int | The maximum number of words to encode into each document vector. |
| 2 | min document frequency | 1 | int | The minimum number of documents a word must appear in to be added to the vocabulary. |
| 3 | max document frequency | PHP_INT_MAX | int | The maximum number of documents a word can appear in to be added to the vocabulary. |
| 4 | tokenizer | Word | Tokenizer | The tokenizer used to extract tokens from blobs of text. |

## Example
```php
use Rubix\ML\Transformers\WordCountVectorizer;
use Rubix\ML\Other\Tokenizers\NGram;

$transformer = new WordCountVectorizer(10000, 3, 500, new NGram(1, 2));
```

## Additional Methods
Return an array of words that comprise each of the vocabularies:
```php
public vocabularies() : array
```
