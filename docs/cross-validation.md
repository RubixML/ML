# Cross Validation
Cross Validation (CV) is a technique for assessing the generalization performance of a model using data it has never seen before. The validation score gives us a sense for how well the model will perform in the real world. In addition, it allows the user to identify problems such as underfitting, overfitting, and selection bias which are discussed in the last section.

## Creating a Testing Set
In some cases we might have a dedicated testing set, but in others we'll need to separate some of the samples from our master dataset to be used for testing. To ensure that both the training and testing sets contain samples that accurately represent the master dataset we have a number of useful methods on the dataset object to employ.

### Randomized Split
The first method of creating a training and testing set that works for all datasets is to randomize and then split the dataset into two subsets of varying proportions. In the example below we'll create a training set with 80% of the samples and a testing set with the remaining 20% using the `randomize()` and `split()` methods on the [Dataset](datasets/api.md) object.

```php
[$training, $testing] = $dataset->randomize()->split(0.8);
```

You can also use the `take()` or `leave()` methods to extract a testing set while leaving the remaining samples in the master dataset.

```php
$testing = $dataset->randomize()->take(1000);
```

### Stratified Split
If we have a [Labeled](datasets/labeled.md) dataset that has categorical labels, we can split the dataset in such a way that samples belonging to each class are represented equally in both sets. This *stratified* method helps to reduce selection bias by ensuring that each subset remains balanced.

```php
[$training, $testing] = $dataset->stratifiedSplit(0.8);
```

## Metrics
Cross validation [Metrics](cross-validation/metrics/api.md) are used to score the predictions made by an [Estimator](estimator.md) with respect to their known ground-truth labels. There are different metrics for different types of problems as shown in the table below.

> **Note:** All metrics follow the schema that higher scores are better - thus, common *loss functions* such as [Mean Squared Error](https://docs.rubixml.com/en/latest/cross-validation/metrics/mean-squared-error.html) and [RMSE](https://docs.rubixml.com/en/latest/cross-validation/metrics/rmse.html) are given as their *negative* to conform to this schema.

### Classification and Anomaly Detection
| Metric | Range | 
|---|---|
| [Accuracy](cross-validation/metrics/accuracy.md) | [0, 1] |
| [F Beta](cross-validation/metrics/f-beta.md) | [0, 1] |
| [Informedness](cross-validation/metrics/informedness.md) | [-1, 1] |
| [MCC](cross-validation/metrics/mcc.md) | [-1, 1] |

### Regression
| Metric | Range | 
|---|---|
| [Mean Absolute Error](cross-validation/metrics/mean-absolute-error.md) | [-∞, 0] |
| [Mean Squared Error](cross-validation/metrics/mean-squared-error.md) | [-∞, 0] |
| [Median Absolute Error](cross-validation/metrics/median-absolute-error.md) | [-∞, 0] |
| [R Squared](cross-validation/metrics/r-squared.md) | [-∞, 1] |
| [RMSE](cross-validation/metrics/rmse.md) | [-∞, 0] |
| [SMAPE](cross-validation/metrics/smape.md) | [-100, 0] |

### Clustering
| Metric | Range | 
|---|---|
| [Completeness](cross-validation/metrics/completeness.md) | [0, 1] |
| [Homogeneity](cross-validation/metrics/homogeneity.md) | [0, 1] |
| [Rand Index](cross-validation/metrics/rand-index.md) | [-1, 1] |
| [V Measure](cross-validation/metrics/v-measure.md) | [0, 1] |

To return a validation score from a Metric pass the predictions and labels to the `score()` method like in the example below.

```php
use Rubix\ML\CrossValidation\Metrics\Accuracy;

$predictions = $estimator->predict($testing);

$metric = new Accuracy();

$score = $metric->score($predictions, $testing->labels());

var_dump($score);
```

```sh
float(0.85)
```

## Reports
Cross validation reports give you a deeper sense for how well a particular model performs with fine-grained information. The `generate()` method on the [Report Generator](cross-validation/reports/api.md#report-generators) interface takes a set of predictions and their corresponding ground-truth labels and returns a [Report](cross-validation/reports/api.md#report-objects) object filled with useful statistics that can be printed directly to the terminal or saved to a file.

### Types of Reports

**Classification and Anomaly Detection**

- [Confusion Matrix](cross-validation/reports/confusion-matrix.md)
- [Multiclass Breakdown](cross-validation/reports/multiclass-breakdown.md)

**Regression**

- [Error Analysis](cross-validation/reports/error-analysis.md)

**Clustering**

- [Contingency Table](cross-validation/reports/contingency-table.md)


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

## Accessing Report Attributes
You can access individual report attributes by treating the report object as an associative array.

```php
$accuracy = $results['accuracy'];
```

### Saving a Report
You may want to save the results of a report for later reference. To do so, you can call the `toJSON()` method on the report object and subsequently the `write()` method on the returned encoding.

```php
$results->toJSON()->write('report.json');
```

## Validators
Metrics can be used stand-alone or they can be used within a [Validator](cross-validation/api.md) object as the scoring function. Validators automate the cross validation process by training and testing a learner on different subsets of a master dataset. The way in which subsets are chosen depends on the algorithm employed under the hood. Most validators implement the [Parallel](parallel.md) interface which allows multiple tests to be run at the same time on multiple CPU cores.

| Validator | Test Coverage | Parallel |
|---|---|---|
| [Hold Out](cross-validation/hold-out.md) | Partial | |
| [K Fold](cross-validation/k-fold.md) | Full | ● |
| [Leave P Out](cross-validation/leave-p-out.md) | Full | ● |
| [Monte Carlo](cross-validation/monte-carlo.md) | Partial | ● |

For example, the K Fold validator automatically selects one of k *folds* of the dataset to use as a validation set and then uses the rest of the folds to train the learner. It will do this until the learner is trained and tested on every sample in the dataset at least once. The final score is then an average of the k validation scores returned by each test. To begin, pass an untrained [Learner](learner.md), a [Labeled](datasets/labeled.md) dataset, and the chosen validation metric to the validator's `test()` method.

```php
use Rubix\ML\CrossValidation\KFold;
use Rubix\ML\CrossValidation\Metrics\FBeta;

$validator = new KFold(10);

$score = $validator->test($estimator, $dataset, new FBeta());

var_dump($score);
```

```sh
float(0.9175)
```

## Common Problems

### Underfitting
A poorly performing model can sometimes be explained as *underfiting* the training data - a condition in which the learner is unable to capture the underlying pattern or trend given the model constraints. Underfitting mostly occurs when a simple model is chosen to represent data that is complex and non-linear. Adding more features can help with underfitting, however if the problem is too severe, a more flexible learner can be chosen for the task instead.

### Overfitting
When a model performs well on training data but poorly during cross validation it could be that the model has *overfit* the training data. Overfitting occurs when the model conforms too closely to the training set data and therefore fails to generalize well to new data or make predictions reliably. Some degree of overfitting can occur with any model, but more flexible models are more prone to overfitting due to their ability to *memorize* individual samples. Most learners employ strategies such as regularization, early stopping, and tree pruning to control overfitting. Adding more samples to the dataset can also help.

### Selection Bias
When a model performs well on certain samples but poorly on others it could be that the learner was trained with a dataset that exhibits selection bias. Selection bias is the bias introduced when a population is disproportionally represented in a dataset. For example, if a learner is trained to classify pictures of cats and dogs with mostly (say 90%) images of cats, it will likely have difficulty in the real world where cats and dogs are more equally represented.