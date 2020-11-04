<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Pipeline.php">[source]</a></span>

# Pipeline
Pipeline is a meta-estimator capable of transforming an input dataset by applying a series of [Transformer](transformers/api.md) *middleware*. Under the hood, Pipeline will automatically fit the training set and transform any [Dataset](datasets/api.md) object supplied as an argument to one of the base estimator's methods before reaching the method context. With *elastic* mode enabled, Pipeline will update the fitting of [Elastic](transformers/api.md#elastic) transformers during partial training.

> **Note:** Since transformations are applied to dataset objects in-place (without making a copy of the data), using a dataset in a program after it has been run through Pipeline may have unexpected results. If you need to keep a *clean* dataset in memory you can clone the dataset object before calling the method on Pipeline that consumes it.

**Interfaces:** [Wrapper](wrapper.md), [Estimator](estimator.md), [Learner](learner.md), [Online](online.md), [Probabilistic](probabilistic.md), [Ranking](ranking.md), [Persistable](persistable.md), [Verbose](verbose.md)

**Data Type Compatibility:** Depends on base learner and transformers

## Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | transformers |  | array | A list of transformers to be applied in order. |
| 2 | estimator |  | Estimator | An instance of a base estimator to receive the transformed data. |
| 3 | elastic | true | bool | Should we update the elastic transformers during partial training? |

## Example
```php
use Rubix\ML\Pipeline;
use Rubix\ML\Transformers\MissingDataImputer;
use Rubix\ML\Transformers\OneHotEncoder;
use Rubix\ML\Transformers\PrincipalComponentAnalysis;
use Rubix\ML\Classifiers\SoftmaxClassifier;

$estimator = new Pipeline([
	new MissingDataImputer(),
	new OneHotEncoder(), 
	new PrincipalComponentAnalysis(20),
], new SoftmaxClassifier(128), true);
```

## Additional Methods
This meta-estimator does not have any additional methods.
