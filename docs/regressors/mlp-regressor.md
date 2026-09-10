<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Regressors/MLPRegressor.php">[source]</a></span>

# MLP Regressor

A multilayer feed-forward neural network with a continuous output layer suitable for regression problems. The Multilayer Perceptron regressor is able to handle complex non-linear regression problems by forming higher-order representations of the input features using intermediate user-defined hidden layers. The MLP also has network snapshotting and progress monitoring to ensure that the model achieves the highest validation score per a given training time budget.

!!! note
    If there are not enough training samples to build an internal validation set with the user-specified holdout ratio then progress monitoring will be disabled.

**Interfaces:** [Estimator](../estimator.md), [Learner](../learner.md), [Online](../online.md), [Verbose](../verbose.md), [Persistable](../persistable.md)

**Data Type Compatibility:** Continuous

## Parameters

| # | Name | Default | Type | Description |
| --- | --- | --- | --- | --- |
| 1 | hiddenLayers | | array | An array composing the user-specified hidden layers of the network in order. |
| 2 | batchSize | 128 | int | The number of training samples to process at a time. |
| 3 | gradientAccumulationSteps | 1 | int | The number of gradient accumulation steps before updating the network parameters. Higher values simulate a larger batch size. |
| 4 | optimizer | Adam | Optimizer | The gradient descent optimizer used to update the network parameters. |
| 5 | maxGradientNorm | null | float | The maximum L2 norm of the gradient set. When exceeded all gradients are rescaled proportionally so that the global norm equals the maximum. |
| 6 | epochs | 1000 | int | The maximum number of training epochs. i.e. the number of times to iterate over the entire training set before terminating. |
| 7 | minChange | 1e-4 | float | The minimum change in the training loss necessary to continue training. |
| 8 | evalInterval | 3 | int | The number of epochs to train before evaluating the model using the holdout set. |
| 9 | window | 5 | int | The number of epochs without improvement in the validation score to wait before considering an early stop. |
| 10 | holdOut | 0.1 | float | The proportion of training samples to use for internal validation. Set to 0 to disable. |
| 11 | costFn | LeastSquares | RegressionLoss | The function that computes the loss associated with an erroneous activation during training. |
| 12 | metric | RMSE | Metric | The metric used to score the generalization performance of the model during training. |

## Example

```php
use Rubix\ML\CrossValidation\Metrics\RSquared;
use Rubix\ML\NeuralNet\ActivationFunctions\ReLU;
use Rubix\ML\NeuralNet\CostFunctions\LeastSquares;
use Rubix\ML\NeuralNet\Layers\Activation;
use Rubix\ML\NeuralNet\Layers\Dense;
use Rubix\ML\NeuralNet\Optimizers\RMSProp;
use Rubix\ML\NeuralNet\Optimizers\Schedulers\Constant;
use Rubix\ML\Regressors\MLPRegressor;

$estimator = new MLPRegressor(
	hiddenLayers: [
		new Dense(neurons: 100),
		new Activation(activationFn: new ReLU()),
		new Dense(neurons: 100),
		new Activation(activationFn: new ReLU()),
		new Dense(neurons: 50),
		new Activation(activationFn: new ReLU()),
		new Dense(neurons: 50),
		new Activation(activationFn: new ReLU()),
	],
	batchSize: 128,
	gradientAccumulationSteps: 1,
	optimizer: new RMSProp(scheduler: new Constant(0.001)),
	maxGradientNorm: null,
	epochs: 100,
	minChange: 1e-5,
	evalInterval: 5,
	window: 10,
	holdOut: 0.1,
	costFn: new LeastSquares(),
	metric: new RSquared()
);
```

## Additional Methods

Return an iterable progress table with the steps from the last training session:

```php
public steps() : iterable
```

```php
use Rubix\ML\Extractors\CSV;

$extractor = new CSV('progress.csv', true);

$extractor->export($estimator->steps());
```

Return the validation score for each epoch from the last training session:

```php
public scores() : float[]|null
```

Return the loss for each epoch from the last training session:

```php
public losses() : float[]|null
```

Returns the underlying neural network instance or `null` if untrained:

```php
public network() : Network|null
```

Clean up any leftover state after training. Only do this if you plan to use the model for inference.

```php
public cleanup() : void
```

Set the path of the temporary snapshot file used to store network parameters during training.

```php
public setSnapshotPath(?string $path) : void
```

## References

[^1]: G. E. Hinton. (1989). Connectionist learning procedures.
[^2]: L. Prechelt. (1997). Early Stopping - but when?
[^3]: R. Pascanu, et al. (2013). On the difficulty of training recurrent neural networks.
