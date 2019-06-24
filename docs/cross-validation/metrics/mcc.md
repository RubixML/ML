<p><span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/CrossValidation/Metrics/MCC.php">Source</a></span></p>

# MCC
Matthews Correlation Coefficient measures the quality of a classification. It takes into account true and false positives and negatives and is generally regarded as a balanced measure which can be used even if the classes are of very different sizes. The MCC is in essence a correlation coefficient between the observed and predicted binary classifications; it returns a value between −1 and +1. A coefficient of +1 represents a perfect prediction, 0 no better than random prediction and −1 indicates total disagreement between prediction and observation. [[Source]](https://github.com/RubixML/RubixML/blob/master/src/CrossValidation/Metrics/MCC.php)

**Estimator Compatibility:** Classifier, Anomaly Detector

**Output Range:** -1 to 1

### Example
```php
use Rubix\ML\CrossValidation\Metrics\MCC;

$metric = new MCC();
```

### References
>- B. W. Matthews. (1975). Decision of the Predicted and Observed Secondary Structure of T4 Phage Lysozyme.