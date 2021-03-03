<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/AnomalyDetectors/OneClassSVM.php">[source]</a></span>

# One Class SVM
An unsupervised Support Vector Machine (SVM) used for anomaly detection. The One Class SVM aims to find a maximum margin between a set of data points and the *origin*, rather than between classes such as with [SVC](../classifiers/svc.md).

!!! note
    This estimator requires the [SVM extension](https://php.net/manual/en/book.svm.php) which uses the libsvm engine under the hood.

**Interfaces:** [Estimator](../estimator.md), [Learner](../learner.md)

**Data Type Compatibility:** Continuous

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | nu | 0.1 | float | An upper bound on the percentage of margin errors and a lower bound on the percentage of support vectors. |
| 2 | kernel | RBF | Kernel | The kernel function used to express non-linear data in higher dimensions. |
| 3 | shrinking | true | bool | Should we use the shrinking heuristic? |
| 4 | tolerance | 1e-3 | float | The minimum change in the cost function necessary to continue training. |
| 5 | cacheSize | 100.0 | float | The size of the kernel cache in MB. |

## Example
```php
use Rubix\ML\AnomalyDetectors\OneClassSVM;
use Rubix\ML\Kernels\SVM\Polynomial;

$estimator = new OneClassSVM(0.1, new Polynomial(4), true, 1e-3, 100.0);
```

## Additional Methods
Save the model data to the filesystem:
```php
public save(string $path) : void
```

Load the model data from the filesystem:
```php
public load(string $path) : void
```

## References
[^1]: C. Chang et al. (2011). LIBSVM: A library for support vector machines.