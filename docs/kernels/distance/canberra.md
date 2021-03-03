<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Kernels/Distance/Canberra.php">[source]</a></span>

# Canberra
A weighted version of the [Manhattan](manhattan.md) distance, Canberra examines the sum of a series of fractional differences between two samples. Canberra can be very sensitive when both coordinates are near zero.

$$
Canberra(\mathbf {a} ,\mathbf {b} )=\sum _{i=1}^{n}{\frac {|a_{i}-b_{i}|}{|a_{i}|+|b_{i}|}}
$$

**Data Type Compatibility:** Continuous

## Parameters
This kernel does not have any parameters.

## Example
```php
use Rubix\ML\Kernels\Distance\Canberra;

$kernel = new Canberra();
```

## References
[^1]: G. N. Lance et al. (1967). Mixed-data classificatory programs I. Agglomerative Systems.
