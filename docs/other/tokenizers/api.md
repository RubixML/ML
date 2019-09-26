# Tokenizers
Tokenizers take a body of text and convert the words to an array of string *tokens*. Tokens can represent a single word or multiple words such as in [NGram](n-gram.md) and [SkipGram](skip-gram.md). Tokenizers are used by various transformers in Rubix such as the [Word Count Vectorizer](../../transformers/word-count-vectorizer.md) to represent blobs of text as token counts.

### Tokenizing Text
To tokenize a blob of text:
```php
public tokenize(string $text) : array
```

**Example**

```php
use Rubix\ML\Extractors\Tokenizers\Word;

$tokenizer = new Word();

$tokens = $tokenizer->tokenize('I would like to die on Mars, just not on impact.');

var_dump($tokens);
```

```sh
array(10) {
    [0]=> string(5) "would"
	[1]=> string(4) "like"
	[2]=> string(2) "to"
	[3]=> string(3) "die"
	[4]=> string(2) "on"
	[5]=> string(4) "Mars"
	[6]=> string(4) "just"
	[7]=> string(3) "not"
	[8]=> string(2) "on"
	[9]=> string(6) "impact"
}
```