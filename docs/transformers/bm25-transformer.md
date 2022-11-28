<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Transformers/BM25Transformer.php">[source]</a></span>

# BM25 Transformer
BM25 is a sublinear term weighting scheme that takes term frequency (TF), document frequency (DF), and document length into account. It is similar to [TF-IDF](tf-idf-transformer.md) but with variable sublinearity and the addition of document length normalization.

> **Note:** BM25 Transformer assumes that its inputs are token frequency vectors such as those created by [Word Count Vectorizer](word-count-vectorizer.md).

**Interfaces:** [Transformer](api.md#transformer), [Stateful](api.md#stateful), [Elastic](api.md#elastic)

**Data Type Compatibility:** Continuous only

## Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | dampening | 1.2 | float | The term frequency (TF) dampening factor i.e. the `K1` parameter in the formula. Lower values will cause the TF to saturate quicker. |
| 2 | normalization | 0.75 | float | The importance of document length in normalizing the term frequency i.e. the `b` parameter in the formula. |

## Example
```php
use Rubix\ML\Transformers\BM25Transformer;

$transformer = new BM25Transformer(1.2, 0.75);
```

## Additional Methods
Return the document frequencies calculated during fitting:
```php
public dfs() : ?array
```

Return the average number of tokens per document:
```php
public averageDocumentLength() : ?float
```

### References
>- S. Robertson et al. (2009). The Probabilistic Relevance Framework: BM25 and Beyond.
>- K. Sparck Jones et al. (2000). A probabilistic model of information retrieval: development and comparative experiments.
