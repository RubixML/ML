<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Regressors/MLPRegressor.php">[source]</a></span>

# MLP Regressor

A multilayer feed-forward neural network with a continuous output layer suitable for regression problems. The Multilayer Perceptron regressor is able to handle complex non-linear regression problems by forming higher-order representations of the input features using intermediate user-defined hidden layers. The MLP also has network snapshotting and progress monitoring to ensure that the model achieves the highest validation score per a given training time budget.

!!! note
    Progress monitoring and early stopping require a validation set. Use `setValidationDataset()` to supply one, in which case the learner trains on all of the data passed to `train()`. Without a validation set, `scores()` remains empty and early stopping is disabled.

**Interfaces:** [Estimator](../estimator.md), [Learner](../learner.md), [Iterative](../iterative.md), [Online](../online.md), [Verbose](../verbose.md), [Persistable](../persistable.md)

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
| 7 | minChange | 1e-5 | float | The minimum change in the training loss necessary to continue training. |
| 8 | evalInterval | 1 | int | The number of epochs to train before evaluating the model using the validation set. |
| 9 | window | 10 | int | The number of evaluations without improvement in the validation score to wait before considering an early stop. Set to 0 to disable early stopping. |
| 10 | costFn | LeastSquares | RegressionLoss | The function that computes the loss associated with an erroneous activation during training. |
| 11 | metric | RMSE | Metric | The metric used to score the generalization performance of the model during training. |

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
	costFn: new LeastSquares(),
	metric: new RSquared()
);
```

## Additional Methods

Return the validation score for each epoch from the last training session.

```php
public scores() : float[]|null
```

Return the loss for each epoch from the last training session.

```php
public losses() : float[]|null
```

Return the gradient norm for each epoch from the last training session.

```php
public norms() : float[]|null
```

Returns the underlying neural network instance or `null` if untrained. See [FeedForward](../neural-network/feed-forward.md) for more details.

```php
public network() : FeedForward|null
```

Set the path of the temporary snapshot file used to store network parameters during training.

```php
public setSnapshotPath(?string $path) : void
```

Set the dataset used to score the model during training. The learner always trains on the *entire* dataset given to `train()`. Once a validation dataset is set, `evalInterval` and `window` determine how often it is scored and when training stops early. Pass `null` to disable progress monitoring and early stopping. The dataset is not persisted with the model, so a learner restored from a snapshot must have it set again before resuming.

```php
public setValidationDataset(?Labeled $dataset) : void
```

Clean up any leftover state after training. Only do this if you plan to use the model for inference.

```php
public cleanup() : void
```

## References

[^1]: G. E. Hinton. (1989). Connectionist learning procedures.
[^2]: L. Prechelt. (1997). Early Stopping - but when?
[^3]: R. Pascanu, et al. (2013). On the difficulty of training recurrent neural networks.
