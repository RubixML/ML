<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/NeuralNet/ActivationFunctions/ReLU.php">[source]</a></span>

# ReLU
Rectified Linear Units (ReLU) only output the positive signal of the input. They have the benefit of having a monotonic derivative and are cheap to compute.

## Parameters
This activation function does not have any parameters.

## Example
```php
use Rubix\ML\NeuralNet\ActivationFunctions\ReLU;

$activationFunction = new ReLU(0.1);
```

### References
[^1]: A. L. Maas et al. (2013). Rectifier Nonlinearities Improve Neural Network Acoustic Models.
[^2]: K. Konda et al. (2015). Zero-bias Autoencoders and the Benefits of Co-adapting Features.