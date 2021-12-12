<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Transformers/WordCountVectorizer.php">[source]</a></span>

# Word Count Vectorizer
The Word Count Vectorizer builds a vocabulary from the training samples and transforms text blobs into fixed length sparse feature vectors. Each feature column represents a word or *token* from the vocabulary and the value denotes the number of times that word appears in a given document.

**Interfaces:** [Transformer](api.md#transformer), [Stateful](api.md#stateful), [Persistable](../persistable.md)

**Data Type Compatibility:** Categorical

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | maxVocabularySize | PHP_INT_MAX | int | The maximum number of unique tokens to embed into each document vector. |
| 2 | minDocumentCount | 1 | float | The minimum number of documents a word must appear in to be added to the vocabulary. |
| 3 | maxDocumentRatio | 0.8 | float | The maximum ratio of documents a word can appear in to be added to the vocabulary. |
| 4 | tokenizer | Word | Tokenizer | The tokenizer used to extract features from blobs of text. |

## Example
```php
use Rubix\ML\Transformers\WordCountVectorizer;
use Rubix\ML\Tokenizers\NGram;

$transformer = new WordCountVectorizer(10000, 5, 0.5, new NGram(1, 2));
```

## Additional Methods
Return an array of words that comprise each of the vocabularies:
```php
public vocabularies() : array
```
