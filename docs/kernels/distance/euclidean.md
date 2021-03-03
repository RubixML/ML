<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Kernels/Distance/Euclidean.php">[source]</a></span>

# Euclidean
The straight line (*bee* line) distance between two points. Euclidean distance has the nice property of being invariant under any rotation.

$$
Euclidean\left(a,b\right) = \sqrt {\sum _{i=1}^{n}  \left( a_{i}-b_{i}\right)^2} 
$$

**Data Type Compatibility:** Continuous

## Parameters
This kernel does not have any parameters.

## Example
```php
use Rubix\ML\Kernels\Distance\Euclidean;

$kernel = new Euclidean();
```

## References
[^1]: J. K. Dixon. (1978). Pattern Recognition with Partly Missing Data.