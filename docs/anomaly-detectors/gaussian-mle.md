<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/AnomalyDetectors/GaussianMLE.php">[source]</a></span>

# Gaussian MLE
The Gaussian Maximum Likelihood Estimator (MLE) is able to spot outliers by computing a probability density function (PDF) over the features assuming they are independently and normally (Gaussian) distributed. Samples that are assigned low probability density are more likely to be outliers.

**Interfaces:** [Estimator](../estimator.md), [Learner](../learner.md), [Online](../online.md), [Scoring](../scoring.md), [Persistable](../persistable.md)

**Data Type Compatibility:** Continuous

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | contamination | 0.1 | float | The proportion of outliers that are assumed to be present in the training set. |
| 2 | smoothing | 1e-9 | float | The amount of epsilon smoothing added to the variance of each feature. |

## Example
```php
use Rubix\ML\AnomalyDetectors\GaussianMLE;

$estimator = new GaussianMLE(0.03, 1e-8);
```

## Additional Methods
Return the column means computed from the training set:
```php
public means() : float[]
```

Return the column variances computed from the training set:
```php
public variances() : float[]
```

## References
[^1]: T. F. Chan et al. (1979). Updating Formulae and a Pairwise Algorithm for Computing Sample Variances.
