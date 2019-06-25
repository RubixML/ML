<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Kernels/Distance/Manhattan.php">Source</a></span>

# Manhattan
A distance metric that constrains movement to horizontal and vertical, similar to navigating the city blocks of Manhattan. An example that used this type of movement is a checkers board.

### Parameters
This kernel does not have any parameters.

### Example
```php
use Rubix\ML\Kernels\Distance\Manhattan;

$kernel = new Manhattan();
```