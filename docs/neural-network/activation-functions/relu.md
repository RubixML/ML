<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/NeuralNet/ActivationFunctions/ReLU.php">Source</a></span>

# ReLU
A Thresholded ReLU (Rectified Linear Unit) only outputs the signal above a user-defined threshold parameter.

### Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | threshold | 0. | float | The input value necessary to trigger an activation. |

### Example
```php
use Rubix\ML\NeuralNet\ActivationFunctions\ReLU;

$activationFunction = new ReLU(0.1);
```

### References
>- A. L. Maas et al. (2013). Rectifier Nonlinearities Improve Neural Network Acoustic Models.
>- K. Konda et al. (2015). Zero-bias Autoencoders and the Benefits of Co-adapting Features.