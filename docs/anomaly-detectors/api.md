# Ranking
A Ranking anomaly detector is one that is able to assign arbitrary anomaly scores to samples in a dataset. The samples can then be sorted by their score and the top k samples can be analyzed further.

### Score a Dataset
To rank the dataset by an artitrary scoring function:
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