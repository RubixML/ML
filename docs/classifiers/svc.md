<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Classifiers/SVC.php">Source</a></span>

# SVC
The multiclass Support Vector Machine (SVM) Classifier is a maximum margin classifier that can efficiently perform non-linear classification by implicitly mapping feature vectors into high dimensional feature space (called the *kernel trick*).

> **Note:** This estimator requires the [SVM extension](https://php.net/manual/en/book.svm.php) which uses the LIBSVM engine under the hood.

**Interfaces:** [Estimator](../estimator.md), [Learner](../learner.md)

**Data Type Compatibility:** Continuous

### Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | c | 1.0 | float | The parameter that defines the width of the margin used to separate the classes. |
| 2 | kernel | RBF | object | The kernel function used to operate in higher dimensions. |
| 3 | shrinking | true | bool | Should we use the shrinking heuristic? |
| 4 | tolerance | 1e-3 | float | The minimum change in the cost function necessary to continue training. |
| 5 | cache size | 100. | float | The size of the kernel cache in MB. |

### Additional Methods
Save the model data to the filesystem:
```php
public save(string $path) : void
```

Load the model data from the filesystem:
```php
public load(string $path) : void
```

### Example
```php
use Rubix\ML\Classifiers\SVC;
use Rubix\ML\Kernels\SVM\Linear;

$estimator = new SVC(1.0, new Linear(), true, 1e-3, 100.);

$estimator->train($dataset);

$estimator->save('svm.model');

// ...

$estimator = new SVC();

$estimator->load('svm.model');

$predictions = $estimator->predict($dataset);
```

### References
>- C. Chang et al. (2011). LIBSVM: A library for support vector machines.