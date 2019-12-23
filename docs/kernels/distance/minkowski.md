<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Kernels/Distance/Minkowski.php">[source]</a></span>

# Minkowski
The Minkowski distance can be considered as a generalization of both the [Euclidean](euclidean.md) and [Manhattan](manhattan.md) distances. When the lambda parameter is set to 1 or 2, the distance is equivalent to Manhattan and Euclidean respectively.

**Data Type Compatibility:** Continuous

## Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | lambda | 3.0 | float | Controls the curvature of the unit circle drawn from a point at a fixed distance. |

## Example
```php
use Rubix\ML\Kernels\Distance\Minkowski;

$kernel = new Minkowski(4.0);
```