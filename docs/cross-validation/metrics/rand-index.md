<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/CrossValidation/Metrics/RandIndex.php">Source</a></span>

# Rand Index
The Adjusted Rand Index is a measure of similarity between the clustering and some ground truth that is adjusted for chance. It considers all pairs of samples that are assigned in the same or different clusters in the predicted and empirical clusterings.

**Estimator Compatibility:** Regressor

**Output Range:** -1 to 1

### Example
```php
use Rubix\ML\CrossValidation\Metrics\RandIndex;

$metric = new RandIndex();
```

### References
>- W. M. Rand. (1971). Objective Criteria for the Evaluation of  Clustering Methods.