<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/NeuralNet/Optimizers/StepDecay/StepDecay.php">[source]</a></span>

# Step Decay
A learning rate decay optimizer that reduces the global learning rate by a factor whenever it reaches a new *floor*. The number of steps needed to reach a new floor is defined by the *steps* hyper-parameter.

## Mathematical formulation
Per step (element-wise), the Step Decay learning rate and update are:

$$
\begin{aligned}
\text{floor} &= \left\lfloor \frac{t}{k} \right\rfloor \\
\eta_t &= \frac{\eta_0}{1 + \text{floor}\cdot \lambda} \\
\Delta\theta_t &= \eta_t\,g_t
\end{aligned}
$$

where:
- $t$ is the current step number,
- $k$ is the number of steps per floor,
- $\eta_0$ is the initial learning rate ($rate$),
- $\lambda$ is the decay factor ($decay$),
- $g_t$ is the current gradient.

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | rate | 0.01 | float | The learning rate that controls the global step size. |
| 2 | steps | 100 | int | The size of every floor in steps. i.e. the number of steps to take before applying another factor of decay. |
| 3 | decay | 1e-3 | float | The factor to decrease the learning rate at each *floor*. |

## Example
```php
use Rubix\ML\NeuralNet\Optimizers\StepDecay\StepDecay;

$optimizer = new StepDecay(0.1, 50, 1e-3);
```
