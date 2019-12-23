<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/NeuralNet/Optimizers/Adam.php">[source]</a></span>

# Adam
Short for *Adaptive Moment Estimation*, the Adam Optimizer combines both Momentum and RMS properties. In addition to storing an exponentially decaying average of past squared gradients like [RMSprop](rms-prop.md), Adam also keeps an exponentially decaying average of past gradients, similar to [Momentum](momentum.md). Whereas Momentum can be seen as a ball running down a slope, Adam behaves like a heavy ball with friction.

## Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | rate | 0.001 | float | The learning rate that controls the global step size. |
| 2 | momentum decay | 0.1 | float | The decay rate of the accumulated velocity. |
| 3 | norm decay | 0.001 | float | The decay rate of the rms property. |

## Example
```php
use Rubix\ML\NeuralNet\Optimizers\Adam;

$optimizer = new Adam(0.0001, 0.1, 0.001);
```

### References
>- D. P. Kingma et al. (2014). Adam: A Method for Stochastic Optimization.