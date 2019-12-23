<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Kernels/Distance/Euclidean.php">[source]</a></span>

# Euclidean
Standard straight line (*bee* line) distance between two points. The Euclidean distance has the nice property of being invariant under any rotation.

**Data Type Compatibility:** Continuous

## Parameters
This kernel does not have any parameters.

## Example
```php
use Rubix\ML\Kernels\Distance\Euclidean;

$kernel = new Euclidean();
```

### References
>- J. K. Dixon. (1978). Pattern Recognition with Partly Missing Data.