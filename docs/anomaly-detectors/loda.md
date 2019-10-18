<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/AnomalyDetectors/Loda.php">Source</a></span>

# Loda
*Lightweight Online Detector of Anomalies* uses a sparse random projection matrix to produce input to an ensemble of unique one dimensional equi-width histograms able to estimate the probability density of an unknown sample. The anomaly score is defined as the negative log likelihood of a sample being an outlier. Thus, samples with low probability density will be given a high anomaly score.

**Interfaces:** [Estimator](../estimator.md), [Learner](../learner.md), [Online](../online.md), [Ranking](../ranking.md), [Persistable](../persistable.md)

**Data Type Compatibility:** Continuous

### Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | estimators | 100 | int | The number of projection/histogram pairs in the ensemble. |
| 2 | bins | Auto | int | The number of equi-width bins for each histogram. |
| 3 | threshold | 10.0 | float | The minimum negative log likelihood to be flagged as an anomaly. |

### Additional Methods
To estimate the number of histogram bins from the number of samples in a dataset:
```php
public static estimateBins(int $n) : int
```

### Example
```php
use Rubix\ML\AnomalyDetection\Loda;

$bins = Loda::estimateBins(1000);

$estimator = new Loda(250, $bins, 3.5); // Automatically choose number of bins

$estimator = new Loda(250, 8, 3.5); // Specifying 8 bins
```

### References
>- T. Pevný. (2015). Loda: Lightweight on-line detector of anomalies.
>- L. Birg´e et al. (2005). How Many Bins Should Be Put In A Regular Histogram.