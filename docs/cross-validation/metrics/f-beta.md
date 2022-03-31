<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/CrossValidation/Metrics/FBeta.php">[source]</a></span>

# F-Beta
A weighted harmonic mean of precision and recall, F-Beta is a both a versatile and balanced metric. The beta parameter controls the weight of precision in the combined score. As beta goes to infinity the score only considers recall, whereas when it goes to 0 it only considers precision. When beta is equal to 1, this metric is called an F1 score.

$$
{\displaystyle F_\beta = (1 + \beta^2) \cdot \frac{\mathrm{precision} \cdot \mathrm{recall}}{(\beta^2 \cdot \mathrm{precision}) + \mathrm{recall}}}
$$

**Estimator Compatibility:** Classifier, Anomaly Detector

**Score Range:** 0 to 1

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | beta | 1.0 | float | The ratio of weight given to precision over recall. |

## Example
```php
use Rubix\ML\CrossValidation\Metrics\FBeta;

$metric = new FBeta(0.7);
```