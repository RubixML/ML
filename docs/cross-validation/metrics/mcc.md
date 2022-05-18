<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/CrossValidation/Metrics/MCC.php">[source]</a></span>

# MCC
Matthews Correlation Coefficient (MCC) measures the quality of a classification by taking true and false positives and negatives into account. It is generally regarded as a balanced measure which can be used even if the classes are of very different sizes. A coefficient of 1 represents a perfect prediction, 0 no better than random prediction, and −1 indicates total disagreement between prediction and observation.

$$
{\displaystyle \mathrm {MCC} = {\frac {\mathrm {TP} \times \mathrm {TN} -\mathrm {FP} \times \mathrm {FN} }{\sqrt {(\mathrm {TP} +\mathrm {FP} )(\mathrm {TP} +\mathrm {FN} )(\mathrm {TN} +\mathrm {FP} )(\mathrm {TN} +\mathrm {FN} )}}}}
$$

**Estimator Compatibility:** Classifier, Anomaly Detector

**Score Range:** -1 to 1

## Parameters
This metric does not have any parameters.

## Example
```php
use Rubix\ML\CrossValidation\Metrics\MCC;

$metric = new MCC();
```

## References
[^1]: B. W. Matthews. (1975). Decision of the Predicted and Observed Secondary Structure of T4 Phage Lysozyme.