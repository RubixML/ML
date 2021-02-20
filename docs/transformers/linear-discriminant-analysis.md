<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Transformers/LinearDiscriminantAnalysis.php">[source]</a></span>

# Linear Discriminant Analysis
Linear Discriminant Analysis (LDA) is a supervised dimensionality reduction technique that selects the most informative features using information in the class labels. More formally, LDA finds a linear combination of features that characterizes or best *discriminates* two or more classes.

**Interfaces:** [Transformer](api.md#transformer), [Stateful](api.md#stateful), [Persistable](../persistable.md)

**Data Type Compatibility:** Continuous only

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | dimensions | | int | The target number of dimensions to project onto. |

## Example
```php
use Rubix\ML\Transformers\LinearDiscriminantAnalysis;

$transformer = new LinearDiscriminantAnalysis(20);
```

## Additional Methods
Return the proportion of information lost due to the transformation:
```php
public lossiness() : ?float
```
