<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Classifiers/MultiLayerPerceptron.php">Source</a></span>

# Multi Layer Perceptron
A multiclass feedforward Neural Network classifier that uses a series of user-defined hidden layers as intermediate computational units. Multiple layers and non-linear activation functions allow the Multi Layer Perceptron to approximate any function given enough training data.

> **Note:** The MLP features progress monitoring which stops training early if it can no longer make progress. It also utilizes snapshotting to make sure that it always has the best settings of the model parameters even if progress began to decline during training.

**Interfaces:** [Estimator](../estimator.md), [Learner](../learner.md), [Online](../online.md), [Probabilistic](../probabilistic.md), [Verbose](../verbose.md), [Persistable](../persistable.md)

**Data Type Compatibility:** Continuous

### Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | hidden | | array | An array composing the hidden layers of the neural network. |
| 2 | batch size | 100 | int | The number of training samples to process at a time. |
| 3 | optimizer | Adam | object | The gradient descent optimizer used to train the underlying network. |
| 4 | alpha | 1e-4 | float | The amount of L2 regularization to apply to the weights of the network. |
| 5 | epochs | 1000 | int | The maximum number of training epochs to execute. |
| 6 | min change | 1e-4 | float | The minimum change in the cost function necessary to continue training. |
| 7 | cost fn | Cross Entropy | object | The function that computes the cost of an erroneous activation during training. |
| 8 | holdout | 0.1 | float | The ratio of samples to hold out for progress monitoring. |
| 9 | window | 3 | int | The number of epochs to consider when determining an early stop. |
| 10 | metric | RSquared | object | The metric used to score the generalization performance of the model during training. |

### Additional Methods
Return the average loss of a sample at each epoch of training:
```php
public steps() : array
```

Return the validation scores at each epoch of training:
```php
public scores() : array
```

Returns the underlying neural network instance or *null* if untrained:
```php
public network() : Network|null
```

### Example
```php
use Rubix\ML\Classifiers\MultiLayerPerceptron;
use Rubix\ML\NeuralNet\Layers\Dense;
use Rubix\ML\NeuralNet\Layers\Dropout;
use Rubix\ML\NeuralNet\Layers\Activation;
use Rubix\ML\NeuralNet\ActivationFunctions\LeakyReLU;
use Rubix\ML\NeuralNet\ActivationFunctions\PReLU;
use Rubix\ML\NeuralNet\Optimizers\Adam;
use Rubix\ML\NeuralNet\CostFunctions\CrossEntropy;
use Rubix\ML\CrossValidation\Metrics\MCC;

$estimator = new MultiLayerPerceptron([
    new Dense(100),
    new Activation(new LeakyReLU()),
    new Dropout(0.3),
    new Dense(100),
    new Activation(new LeakyReLU()),
    new Dropout(0.3),
    new Dense(50),
    new PReLU(),
    new Dense(30),
    new PReLU(),
], 100, new Adam(0.001), 1e-4, 1000, 1e-3, new CrossEntropy(), 0.1, 3, new MCC());
```

### References
>- G. E. Hinton. (1989). Connectionist learning procedures.