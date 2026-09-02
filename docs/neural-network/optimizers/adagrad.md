<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/NeuralNet/Optimizers/AdaGrad/AdaGrad.php">[source]</a></span>

# AdaGrad

Short for *Adaptive Gradient*, the AdaGrad Optimizer speeds up the learning of parameters that do not change often and slows down the learning of parameters that do enjoy heavy activity. Due to AdaGrad's infinitely decaying step size, training may be slow or fail to converge using a low learning rate.

## Mathematical formulation
Per step (element-wise), AdaGrad accumulates the sum of squared gradients and scales the update by the root of this sum:

$$
\begin{aligned}
\mathbf{n}_t &= \mathbf{n}_{t-1} + \mathbf{g}_t^{2} \\
\Delta{\theta}_t &= \alpha\, \frac{\mathbf{g}_t}{\sqrt{\mathbf{n}_t} + \varepsilon}
\end{aligned}
$$

where:
- $t$ is the current step,
- $\alpha$ is the learning rate (`rate`),
- $\mathbf{g}_t$ is the current gradient, and $\mathbf{g}_t^{2}$ denotes element-wise square,
- $\varepsilon$ is a small constant for numerical stability (in the implementation, the denominator is clipped from below by `EPSILON`).

## Parameters

| # | Name | Default | Type | Description |
| --- | --- | --- | --- | --- |
| 1 | rate | 0.01 | float | The learning rate that controls the global step size. |

## Example

```php
use Rubix\ML\NeuralNet\Optimizers\AdaGrad\AdaGrad;

$optimizer = new AdaGrad(0.125);
```

## References

[^1]: J. Duchi et al. (2011). Adaptive Subgradient Methods for Online Learning and Stochastic Optimization.
