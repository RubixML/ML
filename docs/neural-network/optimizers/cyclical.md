<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/NeuralNet/Optimizers/Cyclical.php">[source]</a></span>

# Cyclical
The Cyclical optimizer uses a global learning rate that cycles between the lower and upper bound over a designated period while also decaying the upper bound by a factor at each step. Cyclical learning rates have been shown to help escape bad local minima and saddle points of the gradient.

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | lower | 0.001 | float | The lower bound on the learning rate. |
| 2 | upper | 0.006 | float | The upper bound on the learning rate. |
| 3 | steps | 100 | int | The number of steps in every half cycle. |
| 4 | decay | 0.99994 | float | The exponential decay factor to decrease the learning rate by every step. |

## Example
```php
use Rubix\ML\NeuralNet\Optimizers\Cyclical;

$optimizer = new Cyclical(0.001, 0.005, 1000);
```

## References
[^1]: L. N. Smith. (2017). Cyclical Learning Rates for Training Neural Networks.