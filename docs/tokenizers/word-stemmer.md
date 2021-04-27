<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Tokenizers/WordStemmer.php">[source]</a></span>

# Word Stemmer
Word Stemmer reduces inflected and derived words to their root form using the Snowball method. For example, the sentence "Majority voting is likely foolish" stems to "Major vote is like foolish."

!!! note
    For a complete list of [supported languages](https://github.com/wamania/php-stemmer#languages) you can visit the PHP Stemmer documentation.

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | language | | string | The minimum number of contiguous words to a token. |

## Example
```php
use Rubix\ML\Tokenizers\WordStemmer;

$tokenizer = new WordStemmer('english');
```
