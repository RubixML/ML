<p><span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/NeuralNet/ActivationFunctions/Softmax.php">Source</a></span></p>

# Softmax
The Softmax function is a generalization of the [Sigmoid](#sigmoid) function that *squashes* each activation between 0 and 1 *and* all activations together add up to exactly 1.

### Parameters
This activation function does not have any parameters.

### Example
```php
use Rubix\ML\NeuralNet\ActivationFunctions\Softmax;

$activationFunction = new Softmax();
```