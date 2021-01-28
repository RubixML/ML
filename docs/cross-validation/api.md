# Validator
Validators take an instance of a [Learner](../learner.md), a [Labeled](../datasets/labeled.md) dataset object, and a validation [Metric](metrics/api.md) and return a validation score that measures the generalization performance of the model using one of various cross validation techniques.

!!! note
    There is no need to train the learner beforehand. The validator will automatically train the learner on subsets of the dataset created by the testing algorithm.

### Test a Learner
To train and test a Learner on a dataset and return the validation score:
```php
public test(Learner $estimator, Labeled $dataset, Metric $metric) : float
```

```php
use Rubix\ML\CrossValidation\KFold;
use Rubix\ML\CrossValidation\Metrics\Accuracy;

$validator = new KFold(10);

$score = $validator->test($estimator, $dataset, new Accuracy());

var_dump($score);
```

```sh
float(0.869)
```