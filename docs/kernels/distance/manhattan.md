<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Kernels/Distance/Manhattan.php">[source]</a></span>

# Manhattan
A distance metric that constrains movement to horizontal and vertical, similar to navigating the city blocks of Manhattan. An example of a board game that uses this type of movement is Checkers.

$$
Manhattan(\mathbf {a} ,\mathbf {b})=\|\mathbf {a} -\mathbf {b} \|_{1}=\sum _{i=1}^{n}|a_{i}-b_{i}|
$$

**Data Type Compatibility:** Continuous

## Parameters
This kernel does not have any parameters.

## Example
```php
use Rubix\ML\Kernels\Distance\Manhattan;

$kernel = new Manhattan();
```