<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Datasets/Generators/HalfMoon.php">[source]</a></span>

# Half Moon
Generates a dataset consisting of 2-d samples that form the shape of a half moon when plotted on a scatter plot chart.

**Data Types:** Continuous

**Label Type:** Continuous

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | x | 0.0 | float | The *x* coordinate of the center of the half moon. |
| 2 | y | 0.0 | float | The *y* coordinate of the center of the half moon. |
| 3 | scale | 1.0 | float | The scaling factor of the half moon. |
| 4 | rotate | 90.0 | float | The amount in degrees to rotate the half moon counterclockwise. |
| 5 | noise | 0.1 | float | The amount of Gaussian noise to add to each data point as a percentage of the scaling factor. |

## Example
```php
use Rubix\ML\Datasets\Generators\HalfMoon;

$generator = new HalfMoon(4.0, 0.0, 6, 180.0, 0.2);
```

## Additional Methods
This generator does not have any additional methods.
