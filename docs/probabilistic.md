# Probabilistic
Estimators that implement the Probabilistic interface have the `proba()` method that returns an array of joint probability estimates for every possible class or cluster number. Probabilities are useful for ascertaining the degree to which the estimator is *certain* about a particular outcome. A value of 1 indicates that the estimator is 100% certain about a particular class or cluster number. Conversely, a value of 0 means that the estimator is 100% certain that it's *not* that class or cluster number. When the probabilities are considered together they are called a *joint* distribution and always sum to 1.

## Predict Probabilities
Return the joint probability estimates from a dataset:
```php
public proba(Dataset $dataset) : array
```

```php
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
