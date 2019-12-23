<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Transformers/WordCountVectorizer.php">[source]</a></span>

# Word Count Vectorizer
The Word Count Vectorizer builds a vocabulary from the training samples and transforms text blobs into fixed length feature vectors. Each feature column represents a word or *token* from the vocabulary and the value denotes the number of times that word appears in a given sample.

**Interfaces:** [Transformer](api.md#transformer), [Stateful](api.md#stateful)

**Data Type Compatibility:** Categorical

## Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | max vocabulary | PHP_INT_MAX | int | The maximum number of words to encode into each document vector. |
| 2 | min document frequency | 1 | int | The minimum number of documents a word must appear in to be added to the vocabulary. |
| 3 | tokenizer | Word | Tokenizer | The tokenizer that extracts individual words from samples of text. |

## Additional Methods
Return the fitted vocabulary i.e. the words that will be vectorized:
```php
public vocabulary() : array
```

Return the size of the vocabulary:
```php
public size() : int
```

## Example
```php
use Rubix\ML\Transformers\WordCountVectorizer;
use Rubix\ML\Other\Tokenizers\SkipGram;

$transformer = new WordCountVectorizer(10000, 3, new SkipGram());
```