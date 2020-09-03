<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Transformers/LinearDiscriminantAnalysis.php">[source]</a></span>

# Linear Discriminant Analysis
Linear Discriminant Analysis (LDA) is a supervised dimensionality reduction technique that selects the most informative features using information in the class labels. More formally, LDA finds a linear combination of features that characterizes or best *discriminates* two or more classes.

**Interfaces:** [Transformer](api.md#transformer), [Stateful](api.md#stateful)

**Data Type Compatibility:** Continuous only

## Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | dimensions | | int | The target number of dimensions to project onto. |

## Example
```php
use Rubix\ML\Transformers\LinearDiscriminantAnalysis;

$transformer = new LinearDiscriminantAnalysis(20);
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
