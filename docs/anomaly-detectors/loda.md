<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/AnomalyDetectors/Loda.php">[source]</a></span>

# Loda
*Lightweight Online Detector of Anomalies* uses a collection of sparse random projection vectors to provide scalar inputs to an ensemble of unique one-dimensional equi-width histograms. Each histogram then estimates the probability density of the unknown sample using a limited feature set. The final predictions are derived from the averaged densities over the entire ensemble.

**Interfaces:** [Estimator](../estimator.md), [Learner](../learner.md), [Online](../online.md), [Scoring](../scoring.md), [Persistable](../persistable.md)

**Data Type Compatibility:** Continuous

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | estimators | 100 | int | The number of projection/histogram pairs in the ensemble. |
| 2 | bins | null | int | The number of equi-width bins for each histogram. If null then will estimate bin count. |
| 3 | contamination | 0.1 | float | The proportion of outliers that are assumed to be present in the training set. |

## Example
```php
use Rubix\ML\AnomalyDetectors\Loda;

$estimator = new Loda(250, 8, 0.01);
```

## Additional Methods
To estimate the number of histogram bins from the number of samples in a dataset:
```php
public static estimateBins(int $n) : int
```

## References
[^1]: T. Pevný. (2015). Loda: Lightweight on-line detector of anomalies.
[^2]: L. Birg´e et al. (2005). How Many Bins Should Be Put In A Regular Histogram.