# Scoring
A Scoring anomaly detector is one that assigns anomaly scores to unknown samples in a dataset. The interface provides the `score()` method which returns a set of scores from the model. Higher scores indicate a greater degree of anomalousness. In addition, samples can be sorted by their anomaly score to identify the top outliers.

## Score a Dataset
Return the anomaly scores assigned to the samples in a dataset:
```php
public score(Dataset $dataset) : array
```

```php
$scores = $estimator->score($dataset);

print_r($scores);
```

```php
Array
(
    [0] => 0.35033
    [1] => 0.40992
    [2] => 1.68153
)
```
