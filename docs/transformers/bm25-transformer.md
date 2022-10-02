<span style="float:right;"><a href="https://github.com/RubixML/Extras/blob/master/src/Transformers/BM25Transformer.php">[source]</a></span>

# BM25 Transformer
BM25 is a sublinear term frequency weighting scheme that takes term frequency (TF) saturation and document length into account.

> **Note:** BM25 Transformer assumes that its inputs are token frequency vectors such as those created by [Word Count Vectorizer](word-count-vectorizer.md).

**Interfaces:** [Transformer](api.md#transformer), [Stateful](api.md#stateful), [Elastic](api.md#elastic)

**Data Type Compatibility:** Continuous only

## Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | alpha | 1.2 | float | The term frequency (TF) saturation factor. Lower values will cause TF to saturate quicker. |
| 2 | beta | 0.75 | float | The importance of document length in normalizing the term frequency. |

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