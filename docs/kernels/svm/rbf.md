### RBF
Non linear radial basis function computes the distance from a centroid or origin.

**Parameters:**

| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | gamma | null | float | The kernel coefficient. |

**Example:**

```php
use Rubix\ML\Kernels\SVM\RBF;

$kernel = new RBF(null);
```