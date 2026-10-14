<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Kernels/Distance/SquaredEuclidean.php">[source]</a></span>

# Squared Euclidean

The squared Euclidean distance between two points, computed as the sum of the squared differences of their features without the final square root applied by [Euclidean](euclidean.md). This distance is not a metric — it fails the triangle inequality — but it is symmetric and coordinatewise monotone. It is the natural cost function for [Ward](../../clusterers/ward.md) agglomerative clustering.

$$
SquaredEuclidean\left(a,b\right) = \sum _{i=1}^{n}  \left( a_{i}-b_{i}\right)^2
$$

**Interfaces:** Monotonic, Symmetric

**Data Type Compatibility:** Continuous

## Parameters

This kernel does not have any parameters.

## Example

```php
use Rubix\ML\Kernels\Distance\SquaredEuclidean;

$kernel = new SquaredEuclidean();
```

## References

[^1]: J. K. Dixon. (1978). Pattern Recognition with Partly Missing Data.
