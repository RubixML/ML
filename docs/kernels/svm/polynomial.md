<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Kernels/SVM/Polynomial.php">[source]</a></span>

# Polynomial
This kernel projects a sample vector using polynomials of the p'th degree.

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | degree | 3 | int | The degree of the polynomial. |
| 2 | gamma | null | float | The kernel coefficient. |
| 3 | coef0 | 0. | float | The independent term. |

## Example
```php
use Rubix\ML\Kernels\SVM\Polynomial;

$kernel = new Polynomial(3, null, 0.);
```
