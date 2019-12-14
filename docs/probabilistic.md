# Probabilistic
Estimators that implement the *Probabilistic* interface have an additional method that returns an array of probability scores of each possible class, cluster, etc. outcome. Probabilities are useful for ascertaining the degree to which the estimator is certain about a particular prediction.

### Predict Probabilities
Return the probability estimates of a prediction:
```php
public proba(Dataset $dataset) : array
```

**Example**
```php
use Rubix\ML\Datasets\Unlabeled;

// Import unknown samples

$dataset = new Unlabeled($samples);

$probabilities = $estimator->proba($dataset);  

var_dump($probabilities);
```

```sh
array(2) {
	[0] => array(2) {
		['monster'] => 0.975,
		['not monster'] => 0.025,
	}
	[1] => array(2) {
		['monster'] => 0.2,
		['not monster'] => 0.8,
	}
	[2] => array(2) {
		['monster'] => 0.6,
		['not monster'] => 0.4,
	}
}
```

### Predict Probabilities of a Single Sample
Predict the probabilities of a single sample and return the result:
```php
public probaSample(array $sample) : array
```

**Example**

```php
$probabilities = $estimator->probaSample($dataset->sample(1));

var_dump($probabilities);
```

```sh
array(2) {
	['monster'] => 0.6,
	['not monster'] => 0.4,
}
```