<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Kernels/SVM/Sigmoidal.php">[source]</a></span>

# Sigmoidal
S shaped nonliearity kernel with output values ranging from -1 to 1.

## Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | gamma | null | float | The kernel coefficient. |
| 2 | coef0 | 0. | float | The independent term. |

## Example
```php
use Rubix\ML\Kernels\SVM\Sigmoidal;

$kernel = new Sigmoidal(null, 0.);
```