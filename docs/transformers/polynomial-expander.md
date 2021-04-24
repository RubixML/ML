<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Transformers/PolynomialExpander.php">[source]</a></span>

# Polynomial Expander
This transformer will generate polynomials up to and including the specified *degree* of each continuous feature. Polynomial expansion is sometimes used to fit data that is non-linear using a linear estimator such as [Ridge](../regressors/ridge.md), [Logistic Regression](../classifiers/logistic-regression.md), or [Softmax Classifier](../classifiers/softmax-classifier.md).

**Interfaces:** [Transformer](api.md#transformer)

**Data Type Compatibility:** Continuous only

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | degree | 2 | int | The degree of the polynomials to generate for each feature. |

## Example
```php
use Rubix\ML\Transformers\PolynomialExpander;

$transformer = new PolynomialExpander(3);
```

## Additional Methods
This transformer does not have any additional methods.
