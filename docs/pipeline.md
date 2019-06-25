<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Pipeline.php">Source</a></span>

# Pipeline
Pipeline is a meta estimator responsible for transforming the input data by applying a series of [transformer](transformers/api.md) middleware. Pipeline accepts a base estimator and a list of transformers to apply to the input data before it is fed to the estimator. Under the hood, Pipeline will automatically fit the training set upon training and transform any [Dataset object](datasets/api.md) supplied as an argument to one of the base Estimator's methods, including `train()` and `predict()`. With the *elastic* mode enabled, Pipeline can update the fitting of certain transformers during online (*partial*) training.

> **Note:** Since transformations are applied to dataset objects in place (without making a copy), using the dataset in a program after it has been run through Pipeline may have unexpected results. If you need a *clean* dataset object to call multiple methods with, you can use the PHP clone syntax to keep an original (untransformed) copy in memory.

**Interfaces:** [Estimator](estimator.md), [Learner](learner.md), [Online](online.md), [Probabilistic](probabilistic.md), [Persistable](persistable.md), [Verbose](verbose.md)

**Data Type Compatibility:** Depends on base learner and transformers

### Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | transformers |  | array | The transformer middleware to be applied to the input data in order. |
| 2 | estimator |  | object | An instance of the base estimator to receive transformed data. |
| 3 | elastic | true | bool | Should we update the elastic transformers during partial training? |

### Additional Methods
This meta estimator does not have any additional methods.

### Example
```php
use Rubix\ML\Pipeline;
use Rubix\ML\Classifiers\SoftmaxClassifier;
use Rubix\ML\NeuralNet\Optimizer\Adam;
use Rubix\ML\Transformers\NumericStringConverter;
use Rubix\ML\Transformers\MissingDataImputer;
use Rubix\ML\Transformers\OneHotEncoder;
use Rubix\ML\Transformers\PrincipalComponentAnalysis;

$estimator = new Pipeline([
    new NumericStringConverter(),
	new MissingDataImputer('?'),
	new OneHotEncoder(), 
	new PrincipalComponentAnalysis(20),
], new SoftmaxClassifier(100, new Adam(0.001)));
```