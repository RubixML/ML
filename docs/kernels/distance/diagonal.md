<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Kernels/Distance/Diagonal.php">[source]</a></span>

# Diagonal
The Diagonal (a.k.a. *Chebyshev*) distance is a measure that constrains movement to horizontal, vertical, and diagonal. An example of a game that uses diagonal movement is chess.

$$
{\displaystyle Diagonal(a,b)=\max _{i}(|a_{i}-b_{i}|)}
$$

**Data Type Compatibility:** Continuous

## Parameters
This kernel does not have any parameters.

## Example
```php
use Rubix\ML\Kernels\Distance\Diagonal;

$kernel = new Diagonal();
```