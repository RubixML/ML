<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/NeuralNet/ActivationFunctions/ThresholdedReLU/ThresholdedReLU.php">[source]</a></span>

# Thresholded ReLU
A version of the [ReLU](relu.md) function that activates only if the input is above some user-specified threshold level.

$$
{\displaystyle ThresholdedReLU = {\begin{aligned}&{\begin{cases}0&{\text{if }}x\leq \theta \\x&{\text{if }}x>\theta\end{cases}}\end{aligned}}}
$$

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | threshold | 1.0 | float | The threshold at which the neuron is activated. |

## Size and Performance
Thresholded ReLU maintains the computational efficiency of standard ReLU while adding a threshold comparison. It requires only a simple comparison operation against the threshold value and conditional assignment. This makes it nearly as efficient as standard ReLU with minimal additional computational overhead. The threshold parameter allows for controlling neuron sparsity, which can be beneficial for reducing overfitting and improving generalization in certain network architectures. By adjusting the threshold, you can fine-tune the balance between network capacity and regularization without significantly impacting computational performance.

## Plots
<img src="../../images/activation-functions/thresholded-relu.png" alt="Thresholded ReLU Function" width="500" height="auto">

<img src="../../images/activation-functions/thresholded-derivative.png" alt="Thresholded ReLU Derivative" width="500" height="auto">

## Example
```php
use Rubix\ML\NeuralNet\ActivationFunctions\ThresholdedReLU;

$activationFunction = new ThresholdedReLU(2.0);
```

## References
[^1]: K. Konda et al. (2015). Zero-bias Autoencoders and the Benefits of Co-adapting Features.
