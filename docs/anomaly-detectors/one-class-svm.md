### One Class SVM
An unsupervised Support Vector Machine used for anomaly detection. The One Class SVM aims to find a maximum margin between a set of data points and the *origin*, rather than between classes such as with  multiclass SVM ([SVC](#svc)).

> **Note**: This estimator requires the [SVM PHP extension](https://php.net/manual/en/book.svm.php) which uses the LIBSVM engine written in C++ under the hood.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/AnomalyDetectors/OneClassSVM.php)

**Interfaces:** [Learner](#learner), [Persistable](#persistable)

**Compatibility:** Continuous

**Parameters:**

| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | nu | 0.1 | float | An upper bound on the percentage of margin errors and a lower bound on the percentage of support vectors. |
| 2 | kernel | RBF | object | The kernel function used to express non-linear data in higher dimensions. |
| 3 | shrinking | true | bool | Should we use the shrinking heuristic? |
| 4 | tolerance | 1e-3 | float | The minimum change in the cost function necessary to continue training. |
| 5 | cache size | 100. | float | The size of the kernel cache in MB. |

**Additional Methods:**

Save the model data to the filesystem:
```php
public save(string $path) : void
```

Load the model data from the filesystem:
```php
public load(string $path) : void
```

**Example:**

```php
use Rubix\ML\AnomalyDetection\OneClassSVM;
use Rubix\ML\Kernels\SVM\Polynomial;

$estimator = new OneClassSVM(0.1, new Polynomial(4), true, 1e-3, 100.);

$estimator->train($dataset);

$estimator->save('svm.model');

// ...

$estimator = new OneClassSVM();

$estimator->load('svm.model');

$predictions = $estimator->predict($dataset);
```

**References:**

>- C. Chang et al. (2011). LIBSVM: A library for support vector machines.