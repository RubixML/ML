<p><span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/NeuralNet/Optimizers/RMSProp.php">Source</a></span></p>

# RMS Prop
An adaptive gradient technique that divides the current gradient over a rolling window of magnitudes of recent gradients.

### Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | rate | 0.001 | float | The learning rate. i.e. the global step size. |
| 2 | decay | 0.1 | float | The decay rate of the rms property. |

### Example
```php
use Rubix\ML\NeuralNet\Optimizers\RMSProp;

$optimizer = new RMSProp(0.01, 0.1);
```

### References
>- T. Tieleman et al. (2012). Lecture 6e rmsprop: Divide the gradient by a running average of its recent magnitude.