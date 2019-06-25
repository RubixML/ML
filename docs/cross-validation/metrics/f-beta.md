<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/CrossValidation/Metrics/FBeta.php">Source</a></span>

# F Beta
A weighted harmonic mean of precision and recall. The beta parameter controls the weight of precision in the combined score. As beta goes to infinity the score only considers recall whereas when it goes to 0 it only considers precision.

> **Note:** Beta equal to 1 is called an F1 score.

**Estimator Compatibility:** Classifier, Anomaly Detector

**Output Range:** 0 to 1

### Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | beta | 1. | float | The weight of precision in the harmonic mean. |

### Example
```php
use Rubix\ML\CrossValidation\Metrics\FBeta;

$metric = new FBeta(0.7);
```