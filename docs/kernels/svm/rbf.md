<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Kernels/SVM/RBF.php">Source</a></span>

# RBF
Non linear radial basis function (RBF) computes the distance from a centroid or origin.

### Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | gamma | null | float | The kernel coefficient. |

### Example
```php
use Rubix\ML\Kernels\SVM\RBF;

$kernel = new RBF(null);
```