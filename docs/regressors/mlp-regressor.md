<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Regressors/MLPRegressor.php">[source]</a></span>

# MLP Regressor
A multilayer feed-forward neural network with a continuous output layer suitable for regression problems. The Multilayer Perceptron regressor is able to handle complex non-linear regression problems by forming higher-order representations of the input features using intermediate user-defined hidden layers. The MLP also has network snapshotting and progress monitoring to ensure that the model achieves the highest validation score per a given training time budget.

!!! note
    If there are not enough training samples to build an internal validation set with the user-specified holdout ratio then progress monitoring will be disabled.

**Interfaces:** [Estimator](../estimator.md), [Learner](../learner.md), [Online](../online.md), [Verbose](../verbose.md), [Persistable](../persistable.md)

**Data Type Compatibility:** Continuous

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | hidden | | array | An array composing the user-specified hidden layers of the network in order. |
| 2 | batchSize | 128 | int | The number of training samples to process at a time. |
| 3 | optimizer | Adam | Optimizer | The gradient descent optimizer used to update the network parameters. |
| 4 | alpha | 1e-4 | float | The amount of L2 regularization applied to the weights of the output layer. |
| 5 | epochs | 1000 | int | The maximum number of training epochs. i.e. the number of times to iterate over the entire training set before terminating. |
| 6 | minChange | 1e-4 | float | The minimum change in the training loss necessary to continue training. |
| 7 | window | 3 | int | The number of epochs without improvement in the validation score to wait before considering an early stop. |
| 8 | holdOut | 0.1 | float | The proportion of training samples to use for progress monitoring. |
| 9 | costFn | LeastSquares | RegressionLoss | The function that computes the loss associated with an erroneous activation during training. |
| 10 | metric | RMSE | Metric | The metric used to score the generalization performance of the model during training. |

## Example
```php
use Rubix\ML\Regressors\MLPRegressor;
use Rubix\ML\NeuralNet\CostFunctions\LeastSquares;
use Rubix\ML\NeuralNet\Layers\Dense;
use Rubix\ML\NeuralNet\Layers\Activation;
use Rubix\ML\NeuralNet\ActivationFunctions\ReLU;
use Rubix\ML\NeuralNet\Optimizers\RMSProp;
use Rubix\ML\CrossValidation\Metrics\RSquared;

$estimator = new MLPRegressor([
	new Dense(100),
	new Activation(new ReLU()),
	new Dense(100),
	new Activation(new ReLU()),
	new Dense(50),
	new Activation(new ReLU()),
	new Dense(50),
	new Activation(new ReLU()),
], 128, new RMSProp(0.001), 1e-3, 100, 1e-5, 3, 0.1, new LeastSquares(), new RSquared());
```

## Additional Methods
Return the validation score at each epoch from the last training session:
```php
public scores() : float[]|null
```

Return the loss at each epoch from the last training session:
```php
public steps() : float[]|null
```

Returns the underlying neural network instance or `null` if untrained:
```php
public network() : Network|null
```

## References
[^1]: G. E. Hinton. (1989). Connectionist learning procedures.
[^2]: L. Prechelt. (1997). Early Stopping - but when?
