<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Kernels/Distance/NaNEuclidean.php">Source</a></span>

# NaN Euclidean
An Euclidean distance metric suitable for samples that may contain NaN (not a number) values i.e. missing data. The NaN Euclidean metric approximates the Euclidean distance function by dropping NaN values and scaling the distance according to the proportion of non-NaNs (in either a or b or both) to compensate.

### Parameters
This kernel does not have any parameters.

### Example
```php
use Rubix\ML\Kernels\Distance\NaNEuclidean;

$kernel = new NaNEuclidean();
```