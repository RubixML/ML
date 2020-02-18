# Cross Validation
Cross Validation (CV) or *out-of-sample* testing is a technique for assessing the generalization performance of a model using data it has never seen before. The validation score gives us a sense for how well the model will perform in the real world. In addition, it allows the user to identify problems such as underfitting, overfitting, and selection bias which are discussed in the last section.

## Metrics
Cross validation [Metrics](cross-validation/metrics/api.md) are used to score the predictions made by an estimator with respect to their known ground-truth labels. There are different metrics for different types of problems as shown in the table below. All metrics follow the schema that higher scores are better - thus, common *loss functions* such as [Mean Squared Error](https://docs.rubixml.com/en/latest/cross-validation/metrics/mean-squared-error.html) and [RMSE](https://docs.rubixml.com/en/latest/cross-validation/metrics/rmse.html) are given as their *negative* to conform to this schema.

| Metric | Range | Use Cases | 
|---|---|---|
| [Accuracy](cross-validation/metrics/accuracy.md) | [0, 1] | Classification |
| [Completeness](cross-validation/metrics/completeness.md) | [0, 1] | Clustering |
| [F Beta](cross-validation/metrics/f-beta.md) | [0, 1] | Classification, Anomaly Detection |
| [Homogeneity](cross-validation/metrics/homogeneity.md) | [0, 1] | Clustering |
| [Informedness](cross-validation/metrics/informedness.md) | [-1, 1] | Classification, Anomaly Detection |
| [MCC](cross-validation/metrics/mcc.md) | [-1, 1] | Classification, Anomaly Detection |
| [Mean Absolute Error](cross-validation/metrics/mean-absolute-error.md) | [-∞, 0] | Regression |
| [Mean Squared Error](cross-validation/metrics/mean-squared-error.md) | [-∞, 0] | Regression |
| [Median Absolute Error](cross-validation/metrics/median-absolute-error.md) | [-∞, 0] | Regression |
| [R Squared](cross-validation/metrics/r-squared.md) | [-∞, 1] | Regression |
| [Rand Index](cross-validation/metrics/rand-index.md) | [-1, 1] | Clustering |
| [RMSE](cross-validation/metrics/rmse.md) | [-∞, 0] | Regression | 
| [SMAPE](cross-validation/metrics/smape.md) | [-100, 0] | Regression |
| [V Measure](cross-validation/metrics/v-measure.md) | [0, 1] | Clustering |

To return a validation score from a Metric pass the predictions and labels to the `score()` method like in the example below.

```php
use Rubix\ML\CrossValidation\Metrics\Accuracy;

$metric = new Accuracy();

$score = $metric->score($predictions, $labels);

var_dump($score);
```

```sh
float(0.85)
```

## Validators
Metrics can be used stand-alone or they can be used within a [Validator](cross-validation/api.md) object as the scoring function. Validators automate the cross validation process by training and testing a learner on different subsets of a master dataset. The way in which subsets are chosen depends on the algorithm employed under the hood.

| Validator | Test Coverage | Parallel |
|---|---|---|
| [Hold Out](cross-validation/hold-out.md) | Partial | |
| [K Fold](cross-validation/k-fold.md) | Full | ● |
| [Leave P Out](cross-validation/leave-p-out.md) | Full | ● |
| [Monte Carlo](cross-validation/monte-carlo.md) | Partial | ● |

For example, the K Fold validator automatically selects one of k *folds* of the dataset to use as a validation set and then uses the rest of the folds to train the learner. It will do this until the learner is trained and tested on every sample in the dataset at least once. To begin cross validation, pass an untrained learner, a labeled dataset, and the chosen validation metric to the Validator's `test()` method.

```php
use Rubix\ML\CrossValidation\KFold;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\CrossValidation\Metrics\FBeta;

$validator = new KFold(10);

$dataset = new Labeled($samples, $labels);

$score = $validator->test($estimator, $dataset, new FBeta());

var_dump($score);
```

```sh
float(0.9175)
```

## Reports
Cross validation [Reports](cross-validation/reports/api.md) give you a deeper sense for how well a particular model performs with finer-grained information than a Metric. The `generate()` method takes a set of predictions and their corresponding ground-truth labels and returns an associative array (i.e. dictionary or map) filled with information. 

| Report | Use Cases | 
|---|---|
| [Confusion Matrix](cross-validation/reports/confusion-matrix.md) | Classification, Anomaly Detection |
| [Contingency Table](cross-validation/reports/contingency-table.md) | Clustering |
| [Multiclass Breakdown](cross-validation/reports/multiclass-breakdown.md) | Classification, Anomaly Detection |
| [Residual Analysis](cross-validation/reports/residual-analysis.md) | Regression |

For example, the [Residual Analysis](cross-validation/reports/residual-analysis.md) report outputs a number of regression metrics in an associative array.

```php
use Rubix\ML\CrossValidation\Reports\ErrorAnalysis;

$report = new ErrorAnalysis();

$result = $report->generate($predictions, $labels);

var_dump($result);
```

```sh
array(18) {
  ["mean_absolute_error"]=> float(0.8)
  ["median_absolute_error"]=> float(1)
  ["mean_squared_error"]=> float(1)
  ["mean_absolute_percentage_error"]=> float(14.020774976657)
  ["rms_error"]=> float(1)
  ["mean_squared_log_error"]=> float(0.019107097505647)
  ["r_squared"]=> float(0.99589305515627)
  ["error_mean"]=> float(-0.2)
  ["error_midrange"]=> float(-0.5)
  ["error_median"]=> float(0)
  ["error_variance"]=> float(0.96)
  ["error_mad"]=> float(1)
  ["error_iqr"]=> float(2)
  ["error_skewness"]=> float(-0.22963966338592)
  ["error_kurtosis"]=> float(-1.0520833333333)
  ["error_min"]=> int(-2)
  ["error_max"]=> int(1)
  ["cardinality"]=> int(10)
}
```

## Common Problems
Here are some common problems that cross validation can help you identify.

### Underfitting
A poorly performing model can sometimes be explained as *underfiting* the training data - a condition in which the learner is unable to capture the underlying pattern or trend given the model constraints. Underfitting mostly occurs when a simple model is chosen to represent data that is complex and non-linear. Adding more features can help with underfitting, however if the problem is too severe, a more flexible learner can be chosen for the task instead.

### Overfitting
When a model performs well on training data but poorly during cross validation it could be that the model has *overfit* the training data. Overfitting occurs when the model conforms too closely to the training set data and therefore fails to generalize well to new data or make predictions reliably. Some degree of overfitting can occur with any model, but more flexible models are more prone to overfitting due to their ability to *memorize* individual samples. Most learners employ strategies such as regularization, early stopping, and/or pruning to control overfitting. Adding more samples to the dataset can also help.

### Selection Bias
When a model performs well on certain samples but poorly on others it could be that the learner was trained with a dataset that exhibits selection bias. Selection bias is the bias introduced when a population is disproportionally represented in a dataset. For example, if a learner is trained to classify pictures of cats and dogs with mostly (say 90%) images of cats, it will likely have difficulty in the real world where cats and dogs are more equally represented.