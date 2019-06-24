<p><span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Kernels/Distance/Diagonal.php">Source</a></span></p>

# Diagonal
The Diagonal (sometimes called *Chebyshev*) distance is a measure that constrains movement to horizontal, vertical, and diagonal from a point. An example that uses Diagonal movement is a chess board.

### Parameters
This kernel does not have any parameters.

### Example
```php
use Rubix\ML\Kernels\Distance\Diagonal;

$kernel = new Diagonal();
```