# Ranking
A Ranking anomaly detector is one that assigns anomaly scores to samples in a dataset. The interface provides the `rank()` method which returns a set of scores from a dataset object.

## Score a Dataset
Return the anomaly scores assigned to the samples in a dataset:
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
  [2]=> float(1.68163357834096)
}
```

## Rank a Single Sample
Return the anomaly score of a single sample:
```php
public rankSample(array $sample) : float
```

**Example**

```php
$score = $estimator->rankSample([0.001, 6.99, 'chicago', 20000]);

var_dump($score);
```

```sh
float(0.39431742584649)
```