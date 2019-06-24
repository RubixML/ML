# Probabilistic
Estimators that implement the *Probabilistic* interface have an additional method that returns an array of probability scores of each possible class, cluster, etc. Probabilities are useful for ascertaining the degree to which the estimator is certain about a particular prediction.

Return the probability estimates of a prediction:
```php
public proba(Dataset $dataset) : array
```

### Example
```php
$probabilities = $estimator->proba($dataset);  

var_dump($probabilities);
```

**Output:**

```sh
array(2) {
	[0] => array(2) {
		['married'] => 0.975,
		['divorced'] => 0.025,
	}
	[1] => array(2) {
		['married'] => 0.200,
		['divorced'] => 0.800,
	}
}
```