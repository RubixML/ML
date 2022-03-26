<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Regressors/SVR.php">[source]</a></span>

# SVR
The Support Vector Machine Regressor (SVR) is a maximum margin algorithm for the purposes of regression. Similarly to the [SVC](../classifiers/svc.md), the model produced by SVR depends only on a subset of the training data, because the cost function for building the model ignores any training data close to the model prediction given by parameter *epsilon*. Thus, the value of epsilon defines a margin of tolerance where no penalty is given to errors.

!!! note
    This estimator requires the [SVM extension](https://php.net/manual/en/book.svm.php) which uses the libsvm engine under the hood.

**Interfaces:** [Estimator](../estimator.md), [Learner](../learner.md)

**Data Type Compatibility:** Continuous

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | c | 1.0 | float | The parameter that defines the width of the margin used to separate the classes. |
| 2 | epsilon | 0.1 | float | Specifies the margin within which no penalty is associated in the training loss. |
| 3 | kernel | RBF | Kernel | The kernel function used to operate in higher dimensions. |
| 4 | shrinking | true | bool | Should we use the shrinking heuristic? |
| 5 | tolerance | 1e-3 | float | The minimum change in the cost function necessary to continue training. |
| 6 | cache size | 100.0 | float | The size of the kernel cache in MB. |

## Additional Methods
Save the model data to the filesystem:
```php
public save(string $path) : void
```

Load the model data from the filesystem:
```php
public load(string $path) : void
```

## Example
```php
use Rubix\ML\Regressors\SVR;
use Rubix\ML\Kernels\SVM\RBF;

$estimator = new SVR(1.0, 0.03, new RBF(), true, 1e-3, 256.0);
```

## References
[^1]: C. Chang et al. (2011). LIBSVM: A library for support vector machines.
[^2]: A. Smola et al. (2003). A Tutorial on Support Vector Regression.
