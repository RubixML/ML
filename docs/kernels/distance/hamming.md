<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Kernels/Distance/Hamming.php">[source]</a></span>

# Hamming
A categorical distance function that measures distance as the number of substitutions necessary to convert one sample to the other.

**Data Type Compatibility:** Categorical

## Parameters
This kernel does not have any parameters.

## Example
```php
use Rubix\ML\Kernels\Distance\Hamming;

$kernel = new Hamming();
```

## References
[^1]: R. W. Hamming. (1950). Error detecting and error correcting codes.