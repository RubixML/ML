<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/AnomalyDetectors/Loda.php">[source]</a></span>

# Loda
*Lightweight Online Detector of Anomalies* uses a series of sparse random projection vectors to produce scalar inputs to an ensemble of unique one-dimensional equi-width histograms. The histograms are then used to estimate the probability density of an unknown sample during inference.

**Interfaces:** [Estimator](../estimator.md), [Learner](../learner.md), [Online](../online.md), [Ranking](../ranking.md), [Persistable](../persistable.md)

**Data Type Compatibility:** Continuous

## Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | estimators | 100 | int | The number of projection/histogram pairs in the ensemble. |
| 2 | bins | null | int | The number of equi-width bins for each histogram. If null then will estimate bin count. |
| 3 | contamination | 0.1 | float | The proportion of outliers that are assumed to be present in the training set. |

## Additional Methods
To estimate the number of histogram bins from the number of samples in a dataset:
```php
public static estimateBins(int $n) : int
```

## Example
```php
use Rubix\ML\AnomalyDetectors\Loda;

$estimator = new Loda(250, 8, 0.01);
```

### References
>- T. Pevný. (2015). Loda: Lightweight on-line detector of anomalies.
>- L. Birg´e et al. (2005). How Many Bins Should Be Put In A Regular Histogram.