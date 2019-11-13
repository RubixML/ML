<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/NeuralNet/Layers/PReLU.php">[source]</a></span>

# PReLU
Parametric Rectified Linear Units are leaky rectifiers whose leakage coefficient is learned during training.

### Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | initializer | Constant | object | The initializer of the leakage parameter. |

### Example
```php
use Rubix\ML\NeuralNet\Layers\PReLU;
use Rubix\ML\NeuralNet\Initializers\Normal;

$layer = new PReLU(new Normal(0.5));
```

### References
>- K. He et al. (2015). Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification.