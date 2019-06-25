<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Regressors/Adaline.php">Source</a></span>

# Adaline
Adaptive Linear Neuron or (*Adaline*) is a type of single layer neural network with a linear output neuron. Training is equivalent to solving [Ridge](ridge.md) regression iteratively using mini batch Gradient Descent.

**Interfaces:** [Estimator](../estimator.md), [Learner](../learner.md), [Online](../online.md), [Verbose](../verbose.md), [Persistable](../persistable.md)

**Data Type Compatibility:** Continuous

### Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | batch size | 100 | int | The number of training samples to process at a time. |
| 2 | optimizer | Adam | object | The gradient descent optimizer used to train the underlying network. |
| 3 | alpha | 1e-4 | float | The amount of L2 regularization to apply to the weights of the network. |
| 4 | epochs | 100 | int | The maximum number of training epochs to execute. |
| 5 | min change | 1e-4 | float | The minimum change in the cost function necessary to continue training. |
| 6 | cost fn | Least Squares | object | The function that computes the cost of an erroneous activation during training. |

### Additional Methods
Return the average loss of a sample at each epoch of training:
```php
public steps() : array
```

Return the underlying neural network instance or *null* if untrained:
```php
public network() : Network|null
```

### Example
```php
use Rubix\ML\Classifers\Adaline;
use Rubix\ML\NeuralNet\Optimizers\Adam;
use Rubix\ML\NeuralNet\CostFunctions\HuberLoss;

$estimator = new Adaline(10, new Adam(0.001), 500, 1e-6, new HuberLoss(2.5));
```