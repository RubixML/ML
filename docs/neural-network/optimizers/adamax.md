<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/NeuralNet/Optimizers/AdaMax/AdaMax.php">[source]</a></span>

# AdaMax
A version of the [Adam](adam.md) optimizer that replaces the RMS property with the infinity norm of the past gradients. As such, AdaMax is generally more suitable for sparse parameter updates and noisy gradients.

## Mathematical formulation
Per step (element-wise), AdaMax maintains an exponentially decaying moving average of the gradient (velocity) and an infinity-norm accumulator of past gradients, and uses them to scale the update:

$$
\begin{aligned}
\mathbf{v}_t &= (1 - \beta_1)\,\mathbf{v}_{t-1} + \beta_1\,\mathbf{g}_t \\
\mathbf{u}_t &= \max\big(\beta_2\,\mathbf{u}_{t-1},\ |\mathbf{g}_t|\big) \\
\Delta{\theta}_t &= \alpha\, \frac{\mathbf{v}_t}{\max(\mathbf{u}_t, \varepsilon)}
\end{aligned}
$$

where:
- $t$ is the current step,
- $\alpha$ is the learning rate (`rate`),
- $\beta_1$ is the momentum decay (`momentumDecay`),
- $\beta_2$ is the norm decay (`normDecay`),
- $\mathbf{g}_t$ is the current gradient and $|\mathbf{g}_t|$ denotes element-wise absolute value,
- $\varepsilon$ is a small constant for numerical stability (in the implementation, the denominator is clipped from below by `EPSILON`).

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | rate | 0.001 | float | The learning rate that controls the global step size. |
| 2 | momentumDecay | 0.1 | float | The decay rate of the accumulated velocity. |
| 3 | normDecay | 0.001 | float | The decay rate of the infinity norm. |

## Example
```php
use Rubix\ML\NeuralNet\Optimizers\AdaMax\AdaMax;

$optimizer = new AdaMax(0.0001, 0.1, 0.001);
```

## References
[^1]: D. P. Kingma et al. (2014). Adam: A Method for Stochastic Optimization.
