<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/NeuralNet/Initializers/Normal.php">[source]</a></span>

# Normal
Generates a random weight matrix from a Gaussian distribution with user-specified standard deviation.

## Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | stddev | 0.05 | float | The standard deviation of the distribution to sample from. |

## Example
```php
use Rubix\ML\NeuralNet\Initializers\Normal;

$initializer = new Normal(0.1);
```