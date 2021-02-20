<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Transformers/TruncatedSVD.php">[source]</a></span>

# Truncated SVD
Truncated Singular Value Decomposition (SVD) is a matrix factorization and dimensionality reduction technique that generalizes eigendecomposition to general matrices. When applied to datasets of term frequency vectors, the technique is called Latent Semantic Analysis (LSA) and computes a statistical model of relationships between words. Truncated SVD can also be used to compress document representations for fast information retrieval and is known as Latent Semantic Indexing (LSI) in this context.

**Interfaces:** [Transformer](api.md#transformer), [Stateful](api.md#stateful), [Persistable](../persistable.md)

**Data Type Compatibility:** Continuous only

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | dimensions | | int | The target number of dimensions to project onto. |

## Example
```php
use Rubix\ML\Transformers\TruncatedSVD;

$transformer = new TruncatedSVD(100);
```

## Additional Methods
Return the proportion of information lost due to the transformation:
```php
public lossiness() : ?float
```

### References
[^1]: P. W. Foltz. (1996) Latent semantic analysis for text-based research.
