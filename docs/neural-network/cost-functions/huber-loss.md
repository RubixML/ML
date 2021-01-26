<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/NeuralNet/CostFunctions/HuberLoss.php">[source]</a></span>

# Huber Loss
The pseudo Huber Loss function transitions between L1 and L2 loss at a given pivot point (defined by *delta*) such that the function becomes more quadratic as the loss decreases. The combination of L1 and L2 losses make Huber more robust to outliers while maintaining smoothness near the minimum.

$$
L_{\delta}=
    \left\{\begin{matrix}
        \frac{1}{2}(y - \hat{y})^{2} & if \left | (y - \hat{y})  \right | < \delta\\
        \delta ((y - \hat{y}) - \frac1 2 \delta) & otherwise
    \end{matrix}\right.
$$

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | delta | 1.0 | float | The pivot point i.e the point where numbers larger will be evaluated with an L1 loss while number smaller will be evaluated with an L2 loss. |

## Example
```php
use Rubix\ML\NeuralNet\CostFunctions\HuberLoss;

$costFunction = new HuberLoss(0.5);
```