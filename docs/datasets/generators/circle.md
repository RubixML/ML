<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Datasets/Generators/Circle.php">[source]</a></span>

# Circle
Creates a dataset of points forming a circle in 2 dimensions. The label of each sample is the random value used to generate the projection measured in degrees.

**Data Types:** Continuous

**Label Type:** Continuous

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | x | 0.0 | float | The *x* coordinate of the center of the circle. |
| 2 | y | 0.0 | float | The *y* coordinate of the center of the circle. |
| 3 | scale | 1.0 | float | The scaling factor of the circle. |
| 4 | noise | 0.1 | float | The amount of Gaussian noise to add to each data point as a ratio of the scaling factor. |

## Example
```php
use Rubix\ML\Datasets\Generators\Circle;

$generator = new Circle(0.0, 0.0, 100, 0.1);
```

## Additional Methods
This generator does not have any additional methods.
