### Activation
Activation layers apply a nonlinear activation function to their inputs.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/NeuralNet/Layers/Activation.php)

**Parameters:**

| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | activation fn | None | object | The function computes the activation of the layer. |

**Example:**

```php
use Rubix\ML\NeuralNet\Layers\Activation;
use Rubix\ML\NeuralNet\ActivationFunctions\ReLU;

$layer = new Activation(new ReLU());
```