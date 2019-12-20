# Ranking
A Ranking anomaly detector is one that is able to assign arbitrary scores to samples in a dataset such that they can be sorted. The top or bottom *k* samples can then be selected for further analysis by a human expert.

### Score a Dataset
Apply an arbitrary unnormalized scoring function over the dataset:
```php
public rank(Dataset $dataset) : array
```

**Example**

```php
$scores = $estimator->rank($dataset);

var_dump($scores);
```

```sh
array(3) {
  [0]=> float(0.35033859096744)
  [1]=> float(0.40992076925443)
  [2]=> float(0.68163357834096)
}
```

### Rank a Single Sample
Return the score given to a single sample:
```php
public rankSample(array $sample) : float
```

**Example**

```php
$score = $estimator->rankSample($dataset->sample(0));

var_dump($score);
```

```sh
float(0.39431742584649)
```