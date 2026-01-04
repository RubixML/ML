<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/NeuralNet/Optimizers/RMSProp/RMSProp.php">[source]</a></span>

# RMS Prop
An adaptive gradient technique that divides the current gradient over a rolling window of magnitudes of recent gradients. Unlike [AdaGrad](adagrad.md), RMS Prop does not suffer from an infinitely decaying step size.

## Mathematical formulation
Per step (element-wise), RMSProp maintains a running average of squared gradients and scales the step by the root-mean-square:

$$
\begin{aligned}
\rho &= 1 - \text{decay}, \quad \eta = \text{rate} \\
\text{Running average:}\quad v_t &= \rho\,v_{t-1} + (1 - \rho)\,g_t^{\,2} \\
\text{Returned step:}\quad \Delta\theta_t &= \frac{\eta\,g_t}{\max\bigl(\sqrt{v_t},\,\varepsilon\bigr)}
\end{aligned}
$$

where:
- $g_t$ is the current gradient,
- $v_t$ is the running average of squared gradients,
- $\rho$ is the averaging coefficient ($1 − decay$),
- $\eta$ is the learning rate ($rate$),
- $\varepsilon$ is a small constant to avoid division by zero (implemented by clipping $\sqrt{v_t}$ to $[ε, +∞)$).

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | rate | 0.001 | float | The learning rate that controls the global step size. |
| 2 | decay | 0.1 | float | The decay rate of the rms property. |

## Example
```php
use Rubix\ML\NeuralNet\Optimizers\RMSProp\RMSProp;

$optimizer = new RMSProp(0.01, 0.1);
```

## References
[^1]: T. Tieleman et al. (2012). Lecture 6e rmsprop: Divide the gradient by a running average of its recent magnitude.
