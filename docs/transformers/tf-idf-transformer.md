<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Transformers/TfIdfTransformer.php">[source]</a></span>

### TF-IDF Transformer
*Term Frequency - Inverse Document Frequency* is a measurement of how important a word is to a document. The TF-IDF value increases proportionally with the number of times a word appears in a document (*TF*) and is offset by the frequency of the word in the corpus (*IDF*).

> **Note:** This transformer assumes that its input is made up of word frequency vectors such as those produced by [Word Count Vectorizer](word-count-vectorizer.md).

**Interfaces:** [Transformer](api.md#transformer), [Stateful](api.md#stateful), [Elastic](api.md#elastic)

**Data Type Compatibility:** Continuous only

## Parameters
This transformer does not have any parameters.

## Additional Methods
Return the document frequencies calculated during fitting:
```php
public dfs() : ?array
```

## Example

```php
use Rubix\ML\Transformers\TfIdfTransformer;

$transformer = new TfIdfTransformer();
```

### References
>- S. Robertson. (2003). Understanding Inverse Document Frequency: On theoretical arguments for IDF.