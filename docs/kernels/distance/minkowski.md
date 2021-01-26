<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Kernels/Distance/Minkowski.php">[source]</a></span>

# Minkowski
The Minkowski distance can be considered as a generalization of both the [Euclidean](euclidean.md) and [Manhattan](manhattan.md) distances. When the lambda parameter is set to 1 or 2, the distance is equivalent to Manhattan and Euclidean respectively.

$$
{\displaystyle Minkowski\left(a,b\right)=\left(\sum _{i=1}^{n}|a_{i}-b_{i}|^{p}\right)^{\frac {1}{p}}}
$$

**Data Type Compatibility:** Continuous

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | lambda | 3.0 | float | Controls the curvature of the unit circle drawn from a point at a fixed distance. |

## Example
```php
use Rubix\ML\Kernels\Distance\Minkowski;

$kernel = new Minkowski(4.0);
```