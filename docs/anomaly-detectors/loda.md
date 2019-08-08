<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/AnomalyDetectors/Loda.php">Source</a></span>

# Loda
*Lightweight Online Detector of Anomalies* uses a sparse random projection matrix to produce input to an ensemble of unique one dimensional equi-width histograms able to estimate the probability density of an unknown sample. Samples with low density will receive a high anomaly score.

**Interfaces:** [Estimator](../estimator.md), [Learner](../learner.md), [Online](../online.md), [Ranking](api.md#ranking), [Persistable](../persistable.md)

**Data Type Compatibility:** Continuous

### Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | bins | Auto | int | The number of equi-width bins for each histogram. |
| 2 | estimators | 100 | int | The number of histograms in the ensemble. |
| 3 | threshold | 10.0 | float | The threshold anomaly score to be flagged as an outlier. |

### Additional Methods
To estimate the number of histogram bins from a dataset:
```php
public static estimateBins($dataset) : int
```

### Example
```php
use Rubix\ML\AnomalyDetection\Loda;

$bins = Loda::estimateBins(1000);

$estimator = new Loda($bins); // Automatically choose bin count

$estimator = new Loda(5, 250, 3.5); // Specifying bins
```

### References
>- T. Pevný. (2015). Loda: Lightweight on-line detector of anomalies.
>- L. Birg´e et al. (2005). How Many Bins Should Be Put In A Regular Histogram.