<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Regressors/MLPRegressor.php">Source</a></span>

# MLP Regressor
A multi layer feedforward neural network with a continuous output layer suitable for regression problems. Like the [Multi Layer Perceptron](../classifiers/multi-layer-perceptron.md) classifier, the MLP Regressor is able to tackle complex non-linear regression problems by forming higher-order representations of the input features using intermediate computational units called *hidden* layers.

> **Note:** The MLP features progress monitoring which stops training early if it can no longer make progress. It also utilizes snapshotting to make sure that it always has the best settings of the model parameters even if progress began to decline during training.

**Interfaces:** [Estimator](../estimator.md), [Learner](../learner.md), [Online](../online.md), [Verbose](../verbose.md), [Persistable](../persistable.md)

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
| 7 | cost fn | LeastSquares | object | The function that computes the cost of an erroneous activation during training. |
| 8 | holdout | 0.1 | float | The ratio of samples to hold out for progress monitoring. |
| 9 | window | 3 | int | The number of epochs without improvement in the validation score to wait before considering an early stop. |
| 10 | metric | RSquared | object | The metric used to score the generalization performance of the model during training. |

### Additional Methods
Return the training loss at each epoch:
```php
public steps() : array
```

Return the validation scores at each epoch:
```php
public scores() : array
```

Returns the underlying neural network instance or *null* if untrained:
```php
public network() : Network|null
```

### Example
```php
use Rubix\ML\Regressors\MLPRegressor;
use Rubix\ML\NeuralNet\Layers\Dense;
use Rubix\ML\NeuralNet\Layers\Activation;
use Rubix\ML\NeuralNet\ActivationFunctions\LeakyReLU;
use Rubix\ML\NeuralNet\Optimizers\RMSProp;
use Rubix\ML\CrossValidation\Metrics\RSquared;

$estimator = new MLPRegressor([
	new Dense(50),
	new Activation(new LeakyReLU(0.1)),
	new Dense(50),
	new Activation(new LeakyReLU(0.1)),
	new Dense(50),
	new Activation(new LeakyReLU(0.1)),
], 256, new RMSProp(0.001), 1e-3, 100, 1e-5, new LeastSquares(), 0.1, 3, new RSquared());
```

### References
>- G. E. Hinton. (1989). Connectionist learning procedures.