<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Datasets/Generators/Blob.php">[source]</a></span>

# Blob
A normally distributed (Gaussian) n-dimensional blob of samples centered at a given vector. The standard deviation can be set for the whole blob or for each feature column independently. When a global standard deviation is used, the resulting blob will be isotropic and will converge asymptotically to a sphere.

**Data Types:** Continuous

**Label Type:** Unlabeled

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | center | [0, 0] | array | An array containing the coordinates of the center of the blob. |
| 2 | stddev | 1.0 | float or array | Either the global standard deviation or an array with the standard deviation on a per feature column basis. |

## Example
```php
use Rubix\ML\Datasets\Generators\Blob;

$generator = new Blob([-1.2, -5.0, 2.6, 0.8, 10.0], 0.25);
```

## Additional Methods
Fit a Blob generator to the samples in a dataset.
```php
public static simulate(Dataset $dataset) : self
```

Return the center coordinates of the Blob.
```php
public center() : array
```
