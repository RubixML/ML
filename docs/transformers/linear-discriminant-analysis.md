<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Transformers/LinearDiscriminantAnalysis.php">Source</a></span>

# Linear Discriminant Analysis
A supervised dimensionality reduction technique that selects the most discriminating features based on class labels. In other words, LDA finds a linear combination of features that characterizes or best separates two or more classes.

**Interfaces:** [Transformer](api.md#transformer), [Stateful](api.md#stateful)

**Data Type Compatibility:** Continuous only

### Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | dimensions | | int | The target number of dimensions to project onto. |

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
use Rubix\ML\Transformers\LinearDiscriminantAnalysis;

$transformer = new LinearDiscriminantAnalysis(20);
```