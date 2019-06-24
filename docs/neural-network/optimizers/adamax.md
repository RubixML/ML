### AdaMax
A version of [Adam](#adam) that replaces the RMS property with the infinity norm of the gradients. 

> [Source](https://github.com/RubixML/RubixML/blob/master/src/NeuralNet/Optimizers/AdaMax.php)

**Parameters:**

| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | rate | 0.001 | float | The learning rate. i.e. the global step size. |
| 2 | momentum decay | 0.1 | float | The decay rate of the accumulated velocity. |
| 3 | norm decay | 0.001 | float | The decay rate of the infinity norm. |

**Example:**

```php
use Rubix\ML\NeuralNet\Optimizers\AdaMax;

$optimizer = new AdaMax(0.0001, 0.1, 0.001);
```

**References:**

>- D. P. Kingma et al. (2014). Adam: A Method for Stochastic Optimization.