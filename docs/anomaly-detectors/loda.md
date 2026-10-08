<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/AnomalyDetectors/Loda.php">[source]</a></span>

# Loda

*Lightweight Online Detector of Anomalies* uses a collection of sparse random projection vectors to provide scalar inputs to an ensemble of unique one-dimensional equi-width histograms. Each histogram then estimates the probability density of the unknown sample using a limited feature set. The final predictions are derived from the averaged densities over the entire ensemble.

**Interfaces:** [Estimator](../estimator.md), [Learner](../learner.md), [Online](../online.md), [Scoring](../scoring.md), [Persistable](../persistable.md)

**Data Type Compatibility:** Continuous

## Parameters

| # | Name | Default | Type | Description |
| --- | --- | --- | --- | --- |
| 1 | contamination | 0.1 | float | The proportion of outliers that are assumed to be present in the training set. |
| 2 | estimators | 100 | int | The number of projection/histogram pairs in the ensemble. |
| 3 | bins | null | int | The number of equi-width bins for each histogram. If null then will estimate bin count. |

## Example

```php
use Rubix\ML\AnomalyDetectors\Loda;

$estimator = new Loda(0.01, 250, 8);
```

## Additional Methods

This estimator does not have any additional methods.

## References

[^1]: T. Pevný. (2015). Loda: Lightweight on-line detector of anomalies.
[^2]: L. Birg´e et al. (2005). How Many Bins Should Be Put In A Regular Histogram.
