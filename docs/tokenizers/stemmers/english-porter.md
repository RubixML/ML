<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Tokenizers/Stemmers/PorterEnglish.php">[source]</a></span>

# Porter English

The Porter English stemmer reduces English words to their root form using the [Porter algorithm](https://tartarus.org/martin/PorterStemmer/index.html). For example, the word "caresses" is stemmed to "caress" and "ponies" is stemmed to "poni."

## Parameters

This stemmer does not have any parameters.

## Example

```php
use Rubix\ML\Tokenizers\Stemmers\PorterEnglish;

$stemmer = new PorterEnglish();

echo $stemmer->stem('caresses');
// caress
```

## References

[^1]: M. F. Porter. (1980). An algorithm for suffix stripping. Program, 14(3), 130-137.
