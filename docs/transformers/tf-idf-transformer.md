<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Transformers/TfIdfTransformer.php">[source]</a></span>

# TF-IDF Transformer
*Term Frequency - Inverse Document Frequency* is a measurement of how important a word is to a document. The TF-IDF value increases with the number of times a word appears in a document (*TF*) and is offset by the frequency of the word in the corpus (*IDF*).

!!! note
    TF-IDF Transformer assumes that its inputs are token frequency vectors such as those created by [Word Count Vectorizer](word-count-vectorizer.md).

**Interfaces:** [Transformer](api.md#transformer), [Stateful](api.md#stateful), [Elastic](api.md#elastic), [Persistable](../persistable.md)

**Data Type Compatibility:** Continuous only

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | smoothing | 1.0 | float | The amount of additive (Laplace) smoothing to add to the IDFs. |
| 2 | dampening | true | bool | Should we apply a sub-linear function to dampen the effect of recurring tokens? |
| 3 | normalize | true | bool | Should we normalize document lengths? |

## Example
```php
use Rubix\ML\Transformers\TfIdfTransformer;

$transformer = new TfIdfTransformer(2.0, false, false);
```

## Additional Methods
Return the document frequencies calculated during fitting:
```php
public dfs() : ?array
```

Return the average length of a document in tokens:
```php
public averageDocumentLength() : ?float
```

## References
[^1]: S. Robertson. (2003). Understanding Inverse Document Frequency: On theoretical arguments for IDF.
[^2]: S. Robertson et al. (2009). The Probabilistic Relevance Framework: BM25 and Beyond.
