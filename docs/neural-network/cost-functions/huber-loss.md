<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/NeuralNet/CostFunctions/HuberLoss.php">Source</a></span>

# Huber Loss
The *pseudo* Huber Loss function transitions between L1 and L2 (Least Squares) loss at a given pivot point (*delta*) such that the function becomes more quadratic as the loss decreases. The combination of L1 and L2 loss makes Huber Loss robust to outliers while maintaining smoothness near the minimum.

### Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | delta | 1. | float | The pivot point i.e the point where numbers larger will be evaluated with an L1 loss while number smaller will be evaluated with an L2 loss. |

### Example
```php
use Rubix\ML\NeuralNet\CostFunctions\HuberLoss;

$costFunction = new HuberLoss(0.5);
```