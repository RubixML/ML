<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Kernels/Distance/Jaccard.php">Source</a></span>

# Jaccard
This *generalized* Jaccard distance is a measure of distance with a range from 0 to 1 and is thought of as the size of the intersection divided by the size of the union of the two points.

### Parameters
This kernel does not have any parameters.

### Example
```php
use Rubix\ML\Kernels\Distance\Jaccard;

$kernel = new Jaccard();
```