<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Transformers/PrincipalComponentAnalysis.php">[source]</a></span>

# Principal Component Analysis
Principal Component Analysis (PCA) is a dimensionality reduction technique that aims to transform the feature space by the *k* principal components that explain the most variance. PCA is used to compress high-dimensional samples down to lower dimensions such that they would retain as much information as possible.

**Interfaces:** [Transformer](api.md#transformer), [Stateful](api.md#stateful)

**Data Type Compatibility:** Continuous only

## Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | dimensions | | int | The target number of dimensions to project onto. |

## Example
```php
use Rubix\ML\Transformers\PrincipalComponentAnalysis;

$transformer = new PrincipalComponentAnalysis(15);
```

## Additional Methods
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

### References
>- H. Abdi et al. (2010). Principal Component Analysis.