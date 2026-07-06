<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/NeuralNet/Optimizers/Momentum.php">[source]</a></span>

# Momentum
Momentum accelerates each update step by accumulating velocity from past updates and adding a factor of the previous velocity to the current step. Momentum can help speed up training and escape bad local minima when compared with [Stochastic](stochastic.md) Gradient Descent.

## Mathematical formulation
Per step (element-wise), Momentum updates the velocity and applies it as the parameter step:

$$
\begin{aligned}
\beta &= 1 - \text{decay}, \quad \eta = \text{rate} \\
\text{Velocity update:}\quad v_t &= \beta\,v_{t-1} + \eta\,g_t \\
\text{Returned step:}\quad \Delta\theta_t &= v_t
\end{aligned}
$$

Nesterov lookahead (when `lookahead = true`) is approximated by applying the velocity update a second time:

$$
\begin{aligned}
v_t &\leftarrow \beta\,v_t + \eta\,g_t
\end{aligned}
$$

where:
- $g_t$ is the current gradient,
- $v_t$ is the velocity (accumulated update),
- $\beta$ is the momentum coefficient ($1 − decay$),
- $\eta$ is the learning rate ($rate$).

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | rate | 0.001 | float | The learning rate that controls the global step size. |
| 2 | decay | 0.1 | float | The decay rate of the accumulated velocity. |
| 3 | lookahead | false | bool | Should we employ Nesterov's lookahead (NAG) when updating the parameters? |

## Example
```php
use Rubix\ML\NeuralNet\Optimizers\Momentum;

$optimizer = new Momentum(0.01, 0.1, true);
```

## References
[^1]: D. E. Rumelhart et al. (1988). Learning representations by back-propagating errors.
[^2]: I. Sutskever et al. (2013). On the importance of initialization and momentum in deep learning.
