<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/NeuralNet/Optimizers/Adam.php">[source]</a></span>

# Adam
Short for *Adaptive Moment Estimation*, the Adam Optimizer combines both Momentum and RMS properties. In addition to storing an exponentially decaying average of past squared gradients like [RMSprop](rms-prop.md), Adam also keeps an exponentially decaying average of past gradients, similar to [Momentum](momentum.md). Whereas Momentum can be seen as a ball running down a slope, Adam behaves like a heavy ball with friction.

## Mathematical formulation
Per step (element-wise), Adam maintains exponentially decaying moving averages of the gradient and its element-wise square and uses them to scale the update:

$$
\begin{aligned}
\mathbf{v}_t &= (1 - \beta_1)\,\mathbf{v}_{t-1} + \beta_1\,\mathbf{g}_t \\
\mathbf{n}_t &= (1 - \beta_2)\,\mathbf{n}_{t-1} + \beta_2\,\mathbf{g}_t^{2} \\
\Delta{\theta}_t &= \alpha\, \frac{\mathbf{v}_t}{\sqrt{\mathbf{n}_t} + \varepsilon}
\end{aligned}
$$

where:
- $t$ is the current step,
- $\alpha$ is the learning rate (`rate`),
- $\beta_1$ is the momentum decay (`momentumDecay`),
- $\beta_2$ is the norm decay (`normDecay`),
- $\mathbf{g}_t$ is the current gradient, and $\mathbf{g}_t^{2}$ denotes element-wise square,
- $\varepsilon$ is a small constant for numerical stability (in the implementation, the denominator is clipped from below by `EPSILON`).

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | rate | 0.001 | float | The learning rate that controls the global step size. |
| 2 | momentumDecay | 0.1 | float | The decay rate of the accumulated velocity. |
| 3 | normDecay | 0.001 | float | The decay rate of the rms property. |

## Example
```php
use Rubix\ML\NeuralNet\Optimizers\Adam;

$optimizer = new Adam(0.0001, 0.1, 0.001);
```

## References
[^1]: D. P. Kingma et al. (2014). Adam: A Method for Stochastic Optimization.
