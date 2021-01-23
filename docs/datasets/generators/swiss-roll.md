<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Datasets/Generators/SwissRoll.php">[source]</a></span>

# Swiss Roll
Generate a non-linear 3-dimensional dataset resembling a *swiss roll* or spiral. The labels are the seeds to the swiss roll transformation.

**Data Types:** Continuous

**Label Type:** Continuous

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | x | 0.0 | float | The *x* coordinate of the center of the swiss roll. |
| 2 | y | 0.0 | float | The *y* coordinate of the center of the swiss roll. |
| 3 | z | 0.0 | float | The *z* coordinate of the center of the swiss roll. |
| 4 | scale | 1.0 | float | The scaling factor of the swiss roll. |
| 5 | depth | 21.0 | float | The depth of the swiss roll i.e the scale of the y axis. |
| 6 | noise | 0.1 | float | The standard deviation of the gaussian noise. |

## Example
```php
use Rubix\ML\Datasets\Generators\SwissRoll;

$generator = new SwissRoll(5.5, 1.5, -2.0, 10, 21.0, 0.2);
```

## Additional Methods
This generator does not have any additional methods.
