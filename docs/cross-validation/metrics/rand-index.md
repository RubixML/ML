<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/CrossValidation/Metrics/RandIndex.php">[source]</a></span>

# Rand Index
The Adjusted Rand Index is a measure of similarity between a clustering and some ground-truth that is adjusted for chance. It considers all pairs of samples that are assigned in the same or different clusters in the predicted and empirical clusterings.

$$
{\displaystyle ARI = {\frac {\left.\sum _{ij}{\binom {n_{ij}}{2}}-\left[\sum _{i}{\binom {a_{i}}{2}}\sum _{j}{\binom {b_{j}}{2}}\right]\right/{\binom {n}{2}}}{\left.{\frac {1}{2}}\left[\sum _{i}{\binom {a_{i}}{2}}+\sum _{j}{\binom {b_{j}}{2}}\right]-\left[\sum _{i}{\binom {a_{i}}{2}}\sum _{j}{\binom {b_{j}}{2}}\right]\right/{\binom {n}{2}}}}}
$$

**Estimator Compatibility:** Regressor

**Output Range:** -1 to 1

## Parameters
This metric does not have any parameters.

## Example
```php
use Rubix\ML\CrossValidation\Metrics\RandIndex;

$metric = new RandIndex();
```

## References
[^1]: W. M. Rand. (1971). Objective Criteria for the Evaluation of  Clustering Methods.