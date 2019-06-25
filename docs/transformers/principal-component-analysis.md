<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Transformers/PrincipalComponentAnalysis.php">Source</a></span>

# Principal Component Analysis
Principal Component Analysis or *PCA* is a dimensionality reduction technique that aims to transform the feature space by the k *principal components* that explain the most variance of the data where *k* is the dimensionality of the output specified by the user. PCA is used to compress high dimensional samples down to lower dimensions such that they would retain as much of the information as possible.

**Interfaces:** [Transformer](api.md#transformer), [Stateful](api.md#stateful)

**Data Type Compatibility:** Continuous only

### Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | dimensions | None | int | The target number of dimensions to project onto. |

### Additional Methods
Return the amount of variance that has been preserved by the transformation:
```php
public explainedVar() : ?float
```

Return the amount of variance lost by discarding the noise components:
```php
public noiseVar() : ?float
```

Return the percentage of information lost due to the transformation:
```php
public lossiness() : ?float
```

### Example
```php
use Rubix\ML\Transformers\PrincipalComponentAnalysis;

$transformer = new PrincipalComponentAnalysis(15);
```

### References
>- H. Abdi et al. (2010). Principal Component Analysis.