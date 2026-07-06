<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/NeuralNet/Layers/PReLU.php">[source]</a></span>

# PReLU
Parametric Rectified Linear Units are leaky rectifiers whose *leakage* coefficient is learned during training. Unlike standard [Leaky ReLUs](../activation-functions/leaky-relu.md) whose leakage remains constant, PReLU layers can adjust the leakage to better suite the model on a per node basis.

$$
{\displaystyle PReLU = {\begin{cases}\alpha x&{\text{if }}x<0\\x&{\text{if }}x\geq 0\end{cases}}}
$$

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | initializer | Constant | Initializer | The initializer of the leakage parameter. |

## Example
```php
use Rubix\ML\NeuralNet\Layers\PReLU;
use Rubix\ML\NeuralNet\Initializers\Normal;

$layer = new PReLU(new Normal(0.5));
```

## References
[^1]: K. He et al. (2015). Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification.
