# Ranking
A Ranking anomaly detector is one that is able to assign arbitrary scores to samples in a dataset. The samples can then be sorted by their score. For example, in anomaly detection, the top k samples by score can be selected for further analysis.

### Score a Dataset
Apply an arbitrary unnormalized scoring function over the dataset such that the rows can be sorted according to the value:
```php
public rank(Dataset $dataset) : array
```

**Example**

```php
$scores = $estimator->rank($dataset);

var_dump($scores);
```

**Output**

```sh
array(3) {
  [0]=> float(1.80)
  [1]=> int(1.25)
  [2]=> int(9.45)
}
```