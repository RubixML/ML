<p><span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/NeuralNet/Optimizers/Momentum.php">Source</a></span></p>

# Momentum
Momentum adds velocity to each step until exhausted. It does so by accumulating momentum from past updates and adding a factor of the previous velocity to the current step.

### Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | rate | 0.001 | float | The learning rate. i.e. the global step size. |
| 2 | decay | 0.1 | float | The decay rate of the accumulated velocity. |

### Example
```php
use Rubix\ML\NeuralNet\Optimizers\Momentum;

$optimizer = new Momentum(0.001, 0.2);
```

### References
>- D. E. Rumelhart et al. (1988). Learning representations by back-propagating errors.