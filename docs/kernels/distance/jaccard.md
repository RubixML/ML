<p><span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Kernels/Distance/Jaccard.php">Source</a></span></p>

# Jaccard
This *generalized* Jaccard distance is a measure of similarity that one sample has to another with a range from 0 to 1. The higher the percentage, the more dissimilar they are.

### Parameters
This kernel does not have any parameters.

### Example
```php
use Rubix\ML\Kernels\Distance\Jaccard;

$kernel = new Jaccard();
```