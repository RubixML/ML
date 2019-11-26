<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Pipeline.php">[source]</a></span>

# Pipeline
Pipeline is a meta estimator responsible for transforming the input dataset by applying a series of [transformer](transformers/api.md) middleware. Under the hood, Pipeline will automatically fit the training set upon training and transform any [Dataset](datasets/api.md) object supplied as an argument to one of the base estimator's methods, including `train()` and `predict()`. With *elastic* mode enabled, Pipeline will update the fitting of elastic transformers during online (*partial*) training.

> **Note:** Since transformations are applied to dataset objects in-place (without copying the data), using a dataset in a program after it has been run through Pipeline may have unexpected results. If you need to keep a *clean* dataset in memory you can `clone` the dataset object before calling the method on Pipeline.

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
use Rubix\ML\Transformers\NumericStringConverter;
use Rubix\ML\Transformers\MissingDataImputer;
use Rubix\ML\Transformers\OneHotEncoder;
use Rubix\ML\Transformers\PrincipalComponentAnalysis;
use Rubix\ML\Classifiers\SoftmaxClassifier;

$estimator = new Pipeline([
    new NumericStringConverter(),
	new MissingDataImputer('?'),
	new OneHotEncoder(), 
	new PrincipalComponentAnalysis(20),
], new SoftmaxClassifier());
```