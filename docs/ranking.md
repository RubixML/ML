### Ranking
In the way that a Probabilistic estimator ranks the outcome of a particlar sample as a *normalized* (between 0 and 1) value, Ranking estimators rank the outcome by an *arbitrary* score. The purpose of a Ranking estimator is so that you are able to sort the samples by the output. This is useful in cases such as [Anomaly Detection](#anomaly-detectors) where an analyst can flag the top n outliers by rank for further investigation.

To rank the dataset by an artitrary scoring function:
```php
public rank(Dataset $dataset) : array
```

**Example:**

```php
$scores = $estimator->rank($dataset);

var_dump($scores);
```

**Output:**

```sh
array(3) {
  [0]=> float(1.80)
  [1]=> int(1.25)
  [2]=> int(9.45)
}
```