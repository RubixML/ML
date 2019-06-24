# Training and Prediction
Training is the process of feeding the learning algorithm data so that it can build a model of the problem. A trained model consists of all of the parameters (except hyper-parameters) that are required for the estimator to make predictions. If you try to make predictions using an untrained learner, it will throw an exception.

Passing the Labeled dataset to the instantiated learner, we can train our K Nearest Neighbors classifier like so:
```php
$estimator->train($dataset);
```

We can verify that the learner has been trained by calling the `trained()` method:
```php
var_dump($estimator->trained());
```

**Output:**

```sh
bool(true)
```

For our 100 sample example training set, training should only take a matter of microseconds, but larger datasets with higher dimensionality and fancier learning algorithms can take much longer. Once the estimator has been fully trained, we can now feed in some unknown samples to see what the model predicts.

Turning back to our example problem, suppose that we went out and collected 5 new data points from our friends using the same questions we asked the couples we interviewed for our training set. We could make predictions on whether they will stay married or get divorced by taking their answers as features and running them in an Unlabeled dataset through the trained Estimator's `predict()` method.
```php
use Rubix\ML\Datasets\Unlabeled;

$unknown = [
    [4, 3, 44.2], [2, 2, 16.7], [2, 4, 19.5], [1, 5, 8.6], [3, 3, 55.0],
];

$dataset = new Unlabeled($unknown);

$predictions = $estimator->predict($dataset);

var_dump($predictions);
```

**Output:**

```sh
array(5) {
	[0] => 'married'
	[1] => 'divorced'
	[2] => 'divorced'
	[3] => 'divorced'
	[4] => 'married'
}
```