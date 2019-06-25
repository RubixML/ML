<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/NeuralNet/ActivationFunctions/SoftPlus.php">Source</a></span>

# Soft Plus
A smooth approximation of the ReLU function whose output is constrained to be positive.

### Parameters
This activation function does not have any parameters.

### Example
```php
use Rubix\ML\NeuralNet\ActivationFunctions\SoftPlus;

$activationFunction = new SoftPlus();
```

### References
>- X. Glorot et al. (2011). Deep Sparse Rectifier Neural Networks.