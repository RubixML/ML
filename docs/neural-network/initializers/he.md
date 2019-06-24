<p><span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/NeuralNet/Initializers/He.php">Source</a></span></p>

# He
The He initializer was designed for hidden layers that feed into rectified linear unit layers such as [ReLU](#relu), [Leaky ReLU](#leaky-relu), and [ELU](#elu). It draws from a uniform distribution with limits defined as +/- (6 / (fanIn + fanOut)) ** (1. / sqrt(2)).

### Parameters
This initializer does not have any parameters.

### Example
```php
use Rubix\ML\NeuralNet\Initializers\He;

$initializer = new He();
```

### References
>- K. He et al. (2015). Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification.