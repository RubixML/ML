### Noise
This layer adds random Gaussian noise to the inputs to the layer with a standard deviation given as a parameter. Noise added to neural network activations acts as a regularizer by indirectly adding a penalty to the weights through the cost function in the output layer.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/NeuralNet/Layers/Noise.php)

**Parameters:**

| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | stddev | 0.1 | float | The standard deviation of the gaussian noise to add to the inputs. |

**Example:**

```php
use Rubix\ML\NeuralNet\Layers\Noise;

$layer = new Noise(2.0);
```

**References:**

>- C. Gulcehre et al. (2016). Noisy Activation Functions.