<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Transformers/PolynomialExpander.php">[source]</a></span>

# Polynomial Expander
This transformer will generate polynomials up to and including the specified *degree* of each continuous feature column. Polynomial expansion is sometimes used to fit data that is non-linear using a linear estimator such as [Ridge](#ridge) or [Logistic Regression](#logistic-regression).

**Interfaces:** [Transformer](api.md#transformer)

**Data Type Compatibility:** Continuous only

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | degree | 2 | int | The highest degree polynomial to generate from each feature vector. |

## Example
```php
use Rubix\ML\Transformers\PolynomialExpander;

$transformer = new PolynomialExpander(3);
```

## Additional Methods
This transformer does not have any additional methods.
