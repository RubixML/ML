# Ranking
A Ranking anomaly detector is one that is able to assign arbitrary scores to samples in a dataset. The samples can then be sorted by their score and the top k samples can be selected for further analysis.

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
  [0]=> float(0.35033859096744))
  [1]=> int(0.40992076925443)
  [2]=> int(0.68163357834096))
}
```

### Rank a Single Sample
Return the score given to a single sample:
```php
public rankSample(array $sample) : float
```

**Example**

```php
$score = $estimator->rankSample($dataset[1]);

var_dump($score);
```

```sh
float(0.39431742584649)
```