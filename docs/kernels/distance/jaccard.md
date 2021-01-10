<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Kernels/Distance/Jaccard.php">[source]</a></span>

# Jaccard
The *generalized* Jaccard distance is a measure of distance with a range from 0 to 1 and can be thought of as the size of the intersection divided by the size of the union of two points if they were consisted only of binary random variables.

**Data Type Compatibility:** Continuous

## Parameters
This kernel does not have any parameters.

## Example
```php
use Rubix\ML\Kernels\Distance\Jaccard;

$kernel = new Jaccard();
```