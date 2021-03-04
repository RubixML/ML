# Cross Validation
Cross Validation (CV) is a technique for assessing the generalization performance of a model using data it has never seen before. The validation score gives us a sense for how well the model will perform in the real world. In addition, it allows the user to identify problems such as underfitting, overfitting, and selection bias which are discussed in the last section.

## Creating a Testing Set
For some projects we'll create a dedicated testing set, but in others we can separate some of the samples from our master dataset to be used for testing on the fly. To ensure that both the training and testing sets contain samples that accurately represent the master set we have a number of methods on the [Dataset](datasets/api.md) object we can employ.

### Randomized Split
The first method of creating a training and testing set that works for all datasets is to randomize and then split the dataset into two subsets of varying proportions. In the example below we'll create a training set with 80% of the samples and a testing set with the remaining 20% using the `randomize()` and `split()` methods on the Dataset object.

```php
[$training, $testing] = $dataset->randomize()->split(0.8);
```

You can also use the `take()` method to extract a testing set while leaving the remaining samples in the training set.

```php
$testing = $training->randomize()->take(1000);
```

### Stratified Split
If we have a [Labeled](datasets/labeled.md) dataset containing class labels, we can split the dataset in such a way that samples belonging to each class are represented fairly in both sets. This *stratified* method helps to reduce selection bias by ensuring that each subset remains balanced.

```php
[$training, $testing] = $dataset->stratifiedSplit(0.8);
```

## Metrics
Cross validation [Metrics](cross-validation/metrics/api.md) are used to score the predictions made by an [Estimator](estimator.md) with respect to their known ground-truth labels. There are different metrics for different types of problems. To return a validation score from a Metric pass the predictions and labels to the `score()` method like in the example below.

```php
use Rubix\ML\CrossValidation\Metrics\Accuracy;

$predictions = $estimator->predict($testing);

$metric = new Accuracy();

$score = $metric->score($predictions, $testing->labels());

echo $score;
```

```sh
0.85
```

!!! note
    All metrics follow the schema that higher scores are better - thus, common *loss functions* such as [Mean Squared Error](cross-validation/metrics/mean-squared-error.md) and [RMSE](cross-validation/metrics/rmse.md) are given as their *negative* to conform to this schema.

### Classification and Anomaly Detection
Metrics for classification and anomaly detection (a special case of binary classification) compare class predictions to other categorical labels. Their scores are calculated from the true positive (TP), true negative (TN), false positive (FP), and false negative (FN) counts derived from the confusion matrix between the set of predictions and their ground-truth labels.

| Name | Range | Formula | Notes |
|---|---|---|---|
| [Accuracy](cross-validation/metrics/accuracy.md) | [0, 1] | $\frac{TP}{TP + FP}$ | Not suited for imbalanced datasets |
| [F Beta](cross-validation/metrics/f-beta.md) | [0, 1] | $(1 + \beta^2) \cdot \frac{\mathrm{precision} \cdot \mathrm{recall}}{(\beta^2 \cdot \mathrm{precision}) + \mathrm{recall}}$ | |
| [Informedness](cross-validation/metrics/informedness.md) | [-1, 1] | ${\frac {\text{TP}}{{\text{TP}}+{\text{FN}}}}+{\frac {\text{TP}}{{\text{TN}}+{\text{FP}}}}-1$ | |
| [MCC](cross-validation/metrics/mcc.md) | [-1, 1] | ${\frac {\mathrm {TP} \times \mathrm {TN} -\mathrm {FP} \times \mathrm {FN} }{\sqrt {(\mathrm {TP} +\mathrm {FP} )(\mathrm {TP} +\mathrm {FN} )(\mathrm {TN} +\mathrm {FP} )(\mathrm {TN} +\mathrm {FN} )}}}$ | |

### Regression
Regression metrics output a score based on the error achieved by comparing continuous-valued predictions and their ground-truth labels.

| Name | Range | Formula | Notes |
|---|---|---|---|
| [Mean Absolute Error](cross-validation/metrics/mean-absolute-error.md) | [-∞, 0] | ${\frac {1}{n}}{\sum _{i=1}^{n}\left &verbar; Y_{i}-\hat {Y_{i}}\right &verbar; }$ | Output in same units as predictions |
| [Mean Squared Error](cross-validation/metrics/mean-squared-error.md) | [-∞, 0] | ${\frac {1}{n}}\sum _{i=1}^{n}(Y_{i}-{\hat {Y_{i}}})^{2}$ | Sensitive to outliers |
| [Median Absolute Error](cross-validation/metrics/median-absolute-error.md) | [-∞, 0] | ${\operatorname {median} (&verbar; Y_{i}-{\tilde {Y}} &verbar;)}$ | Robust to outliers |
| [R Squared](cross-validation/metrics/r-squared.md) | [-∞, 1] | $1-{SS_{\rm {res}} \over SS_{\rm {tot}}}$ | |
| [RMSE](cross-validation/metrics/rmse.md) | [-∞, 0] | ${\sqrt{ \frac {1}{n} \sum _{i=1}^{n}(Y_{i}-{\hat {Y_{i}}})^{2}}}$ | Output in same units as predictions |
| [SMAPE](cross-validation/metrics/smape.md) | [-100, 0] | ${\frac {100\%}{n}}\sum _{t=1}^{n}{\frac {\left&verbar;F_{t}-A_{t}\right&verbar;}{(&verbar;A_{t}&verbar;+&verbar;F_{t}&verbar;)/2}}$ | |

### Clustering
Clustering metrics derive their scores from a contingency table which can be thought of as a confusion matrix where the class names of the predictions are unknown.

| Name | Range | Formula | Notes |
|---|---|---|---|
| [Completeness](cross-validation/metrics/completeness.md) | [0, 1] | $1-\frac{H(K, C)}{H(K)}$ | Not suited for hyper-parameter tuning |
| [Homogeneity](cross-validation/metrics/homogeneity.md) | [0, 1] | $1-\frac{H(C, K)}{H(C)}$ | Not suited for hyper-parameter tuning |
| [Rand Index](cross-validation/metrics/rand-index.md) | [-1, 1] | ${\frac {\left.\sum _{ij}{\binom {n_{ij}}{2}}-\left[\sum _{i}{\binom {a_{i}}{2}}\sum _{j}{\binom {b_{j}}{2}}\right]\right/{\binom {n}{2}}}{\left.{\frac {1}{2}}\left[\sum _{i}{\binom {a_{i}}{2}}+\sum _{j}{\binom {b_{j}}{2}}\right]-\left[\sum _{i}{\binom {a_{i}}{2}}\sum _{j}{\binom {b_{j}}{2}}\right]\right/{\binom {n}{2}}}}$ | |
| [V Measure](cross-validation/metrics/v-measure.md) | [0, 1] | $\frac{(1+\beta)hc}{\beta h + c}$ | |

## Reports
Cross validation reports give you a deeper sense for how well a particular model performs with fine-grained information. The `generate()` method on the [Report Generator](cross-validation/reports/api.md#report-generators) interface takes a set of predictions and their corresponding ground-truth labels and returns a [Report](cross-validation/reports/api.md#report-objects) object filled with useful statistics that can be printed directly to the terminal or saved to a file.

| Report | Usage |
|---|---|
| [Confusion Matrix](cross-validation/reports/confusion-matrix.md) | Classification or Anomaly Detection |
| [Contingency Table](cross-validation/reports/contingency-table.md) | Clustering |
| [Error Analysis](cross-validation/reports/error-analysis.md) | Regression |
| [Multiclass Breakdown](cross-validation/reports/multiclass-breakdown.md) | Classification or Anomaly Detection |

### Generating a Report
To generate the report, pass the predictions made by an estimator and their ground-truth labels to the `generate()` method on the report generator instance.

```php
use Rubix\ML\CrossValidation\Reports\ErrorAnalysis;

$report = new ErrorAnalysis();

$results = $report->generate($predictions, $labels);
```

### Printing a Report
The results of the report are returned in a [Report](cross-validation/reports/api.md#report-objects) object. Report objects implement the Stringable interface which means they can be cast to strings to output the human-readable form of the report.

```php
echo $results;
```

```json
{
    "mean_absolute_error": 0.8,
    "median_absolute_error": 1,
    "mean_squared_error": 1,
    "mean_absolute_percentage_error": 14.02077497665733,
    "rms_error": 1,
    "mean_squared_log_error": 0.019107097505647368,
    "r_squared": 0.9958930551562692,
    "error_mean": -0.2,
    "error_midrange": -0.5,
    "error_median": 0,
    "error_variance": 0.9599999999999997,
    "error_mad": 1,
    "error_iqr": 2,
    "error_skewness": -0.22963966338592326,
    "error_kurtosis": -1.0520833333333324,
    "error_min": -2,
    "error_max": 1,
    "cardinality": 10
}.
```

### Accessing Report Attributes
You can access individual report attributes by treating the report object as an associative array.

```php
$mae = $results['mean_absolute_error'];
```

### Saving a Report
You may want to save the results of a report for later reference. To do so, you may call the `toJSON()` method on the report object and subsequently the `write()` method on the returned encoding to save the report to a file.

```php
$results->toJSON()->write('report.json');
```

## Validators
Metrics can be used stand-alone or they can be used within a [Validator](cross-validation/api.md) object as the scoring function. Validators automate the cross validation process by training and testing a learner on different subsets of a master dataset. The way in which subsets are chosen depends on the algorithm employed under the hood. Most validators implement the [Parallel](parallel.md) interface which allows multiple tests to be run at the same time in parallel.

| Validator | Test Coverage | Parallel |
|---|---|---|
| [Hold Out](cross-validation/hold-out.md) | Partial | |
| [K Fold](cross-validation/k-fold.md) | Full | ● |
| [Leave P Out](cross-validation/leave-p-out.md) | Full | ● |
| [Monte Carlo](cross-validation/monte-carlo.md) | Asymptotically Full | ● |

For example, the K Fold validator automatically selects one of k subsets referred to as a *fold* as a validation set and then uses the rest of the folds to train the learner. It does this until the learner is trained and tested on every sample in the dataset at least once. The final score is then an average of the k validation scores returned by each test. To begin, pass an untrained [Learner](learner.md), a [Labeled](datasets/labeled.md) dataset, and your chosen validation metric to the validator's `test()` method.

```php
use Rubix\ML\CrossValidation\KFold;
use Rubix\ML\CrossValidation\Metrics\FBeta;

$validator = new KFold(5);

$score = $validator->test($estimator, $dataset, new FBeta());

echo $score;
```

```sh
0.9175
```

## Common Problems
Poor generalization performance can be explained by one or more of these common problems.

### Underfitting
A poorly performing model can sometimes be explained as *underfitting* the training data - a condition in which the learner is unable to capture the underlying pattern or trend given the model constraints. The result of an underfit model is an estimator with high bias error. Underfitting usually occurs when a simple model is chosen to represent data that is complex and non-linear. Adding more features to the dataset can help, however if the problem is too severe, a more flexible learning algorithm can be chosen for the task instead.

### Overfitting
When a model performs well on training data but poorly during cross-validation it could be that the model has *overfit* the training data. Overfitting occurs when the model conforms too closely to the training data and therefore fails to generalize well to new data or make predictions reliably. Flexible models are more prone to overfitting due to their ability to *memorize* individual samples. Most learners employ strategies such as regularization, early stopping, and/or tree pruning to control overfitting, however if overfitting is still a problem, adding more samples to the dataset can also help.

### Selection Bias
When a model performs well on certain samples but poorly on others it could be that the learner was trained with a dataset that exhibits selection bias. Selection bias is the bias introduced when a population is disproportionally represented in a dataset. For example, if a learner is trained to classify pictures of cats and dogs but mostly (say 90%) cats are represented in the dataset, the model will likely have difficulty making real-world predictions where cats and dogs are more equally represented. To correct selection bias, either obtain more training samples or up-sample the class of the underrepresented type.
