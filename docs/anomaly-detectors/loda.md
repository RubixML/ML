<p><span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/AnomalyDetectors/LODA.php">Source</a></span></p>

# LODA
Lightweight Online Detector of Anomalies uses sparse random projection vectors to produce an ensemble of unique one dimensional equi-width histograms able to estimate the probability density of an unknown sample. The anomaly score is given by the negative log likelihood whose upper threshold can be set by the user.

**Interfaces:** [Estimator](#estimators), [Learner](#learner), [Online](#online), [Ranking](#ranking), [Persistable](#persistable)

**Data Type Compatibility:** Continuous

### Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | bins | Auto | int | The number of equi-width bins for each histogram. |
| 2 | estimators | 100 | int | The number of random projections and histograms. |
| 3 | threshold | 5.5 | float | The threshold anomaly score to be flagged as an outlier. |

### Additional Methods
To estimate the number of histogram bins from a dataset:
```php
public static estimateBins($dataset) : int
```

### Example
```php
use Rubix\ML\AnomalyDetection\LODA;

$estimator = new LODA(5, 250, 3.5);
```

### References
>- T. Pevný. (2015). Loda: Lightweight on-line detector of anamalies.
>- L. Birg´e et al. (2005). How Many Bins Should Be Put In A Regular Histogram.