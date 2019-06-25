# Validators
Validators take an [Estimator](../estimator.md) instance, [Labeled](../datasets/labeled.md) dataset object, and validation [Metric](metrics/api.md) and return a validation score that measures the generalization performance of the model using one of various cross validation techniques. There is no need to train the estimator beforehand as the validator will automatically train it on subsets of the dataset created by the testing algorithm.

### Test an Estimator
```php
public test(Estimator $estimator, Labeled $dataset, Validation $metric) : float
```

**Example**

```php
use Rubix\ML\CrossValidation\KFold;
use Rubix\ML\CrossValidation\Metrics\Accuracy;

$validator = new KFold(10);

$score = $validator->test($estimator, $dataset, new Accuracy());

var_dump($score);
```

**Output**
```sh
float(0.869)
```