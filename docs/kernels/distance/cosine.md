<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Kernels/Distance/Cosine.php">[source]</a></span>

# Cosine
Cosine Similarity is a measure that ignores the magnitude of the distance between two non-zero vectors thus acting as strictly a judgement of orientation. Two vectors with the same orientation have a cosine similarity of 1, whereas two vectors oriented at 90° relative to each other have a similarity of 0, and two vectors diametrically opposed have a similarity of -1. To be used as a distance function, we subtract the Cosine Similarity from 1 in order to satisfy the positive semi-definite condition, therefore the Cosine *distance* is a number between 0 and 2.

> **Note:** This distance kernel is optimized for sparse (mainly zeros) coordinate vectors.

**Data Type Compatibility:** Continuous

## Parameters
This kernel does not have any parameters.

## Example
```php
use Rubix\ML\Kernels\Distance\Cosine;

$kernel = new Cosine();
```