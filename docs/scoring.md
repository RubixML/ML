# Scoring
A Scoring anomaly detector is one that assigns anomaly scores to unknown samples in a dataset. The interface provides the `score()` method which returns a set of scores from the model. Higher scores indicate a greater degree of anomalousness. In addition, samples can be sorted by their anomaly score to identify the top outliers.

## Score a Dataset
Return the anomaly scores assigned to the samples in a dataset:
```php
public score(Dataset $dataset) : array
```

```php
$scores = $estimator->score($dataset);

var_dump($scores);
```

```sh
array(3) {
  [0]=> float(0.35033859096744)
  [1]=> float(0.40992076925443)
  [2]=> float(1.68163357834096)
}
```
