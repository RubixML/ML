<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Kernels/Distance/SafeEuclidean.php">[source]</a></span>

# Safe Euclidean
An Euclidean distance metric suitable for samples that may contain NaN (not a number) values i.e. missing data. The Safe Euclidean metric approximates the Euclidean distance function by dropping NaN values and scaling the distance according to the proportion of non-NaNs (in either a or b or both) to compensate.

**Data Type Compatibility:** Continuous

## Parameters
This kernel does not have any parameters.

## Example
```php
use Rubix\ML\Kernels\Distance\SafeEuclidean;

$kernel = new SafeEuclidean();
```

## References
[^1]: J. K. Dixon. (1978). Pattern Recognition with Partly Missing Data.