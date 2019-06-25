<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Datasets/Generators/Agglomerate.php">Source</a></span>

# Blob
A normally distributed (Gaussian) n-dimensional blob of samples centered at a given vector. The standard deviation can be set for the whole blob or for each feature column independently. When a global value is used, the resulting blob will be isotropic and will converge asypmtotically to a sphere.

**Data Types:** Continuous

**Label Type:** None

### Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | center | [0.0, 0.0] | array | An array containing the coordinates of the center of the blob. |
| 2 | stddev | 1.0 | float or array | Either the global standard deviation or an array with the standard deviation on a per feature column basis. |

### Additional Methods
This generator does not have any additional methods.

### Example
```php
use Rubix\ML\Datasets\Generators\Blob;

$generator = new Blob([-1.2, -5., 2.6, 0.8, 10.], 0.25);
```