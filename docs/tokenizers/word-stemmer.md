<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Tokenizers/WordStemmer.php">[source]</a></span>

# Word Stemmer

The Word Stemmer reduces inflected and derived words to their root form using a [stemmer](stemmers/stemmer.md). For example, the sentence "Majority voting is likely foolish" might stem to "major vote is like foolish."

## Parameters

| # | Name | Default | Type | Description |
| --- | --- | --- | --- | --- |
| 1 | stemmer | PorterEnglish | [Stemmer](stemmers/stemmer.md) | The stemmer algorithm to reduce words to their root form. |

## Example

```php
use Rubix\ML\Tokenizers\WordStemmer;

$tokenizer = new WordStemmer();

var_export($tokenizer->tokenize('Majority voting is likely foolish'));
// ['Major', 'vote', 'is', 'like', 'foolish']
```
