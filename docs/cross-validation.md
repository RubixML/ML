# Cross Validation
Cross Validation (CV) or *out-of-sample* testing is the primary technique for assessing the accuracy of a model using data it has never seen before. The validation score gives us a sense for how well the model will perform in the real world. In addition, it allows the user to identify problems such as underfitting, overfitting, or selection bias.

## Metrics
Cross validation [Metrics](https://docs.rubixml.com/en/latest/cross-validation/metrics/api.html) are used to score the predictions made by an estimator with respect to their ground-truth labels. There are different metrics for different estimator types. For example, one can measure the accuracy of a classifier or the mean squared error (MSE) of a regressor.

| Task | Example Metrics |
|---|---|
| Classification | [Accuracy](https://docs.rubixml.com/en/latest/cross-validation/metrics/accuracy.html), [F Beta](https://docs.rubixml.com/en/latest/cross-validation/metrics/f-beta.html), [MCC](https://docs.rubixml.com/en/latest/cross-validation/metrics/mcc.html), [Informedness](https://docs.rubixml.com/en/latest/cross-validation/metrics/informedness.html) |
| Regression | [Mean Absolute Error](https://docs.rubixml.com/en/latest/cross-validation/metrics/mean-absolute-error.html), [R Squared](https://docs.rubixml.com/en/latest/cross-validation/metrics/r-squared.html), [SMAPE](https://docs.rubixml.com/en/latest/cross-validation/metrics/smape.html) |
| Clustering | [Homogeneity](https://docs.rubixml.com/en/latest/cross-validation/metrics/homogeneity.html), [V Measure](https://docs.rubixml.com/en/latest/cross-validation/metrics/v-measure.html), [Rand Index](https://docs.rubixml.com/en/latest/cross-validation/metrics/rand-index.html) |
| Anomaly Detection | [F Beta](https://docs.rubixml.com/en/latest/cross-validation/metrics/f-beta.html), [MCC](https://docs.rubixml.com/en/latest/cross-validation/metrics/mcc.html) |

**Example**

```php
use Rubix\ML\CrossValidation\Metrics\Accuracy;

// Make predictions and import labels

$metric = new Accuracy();

$score = $metric->score($predictions, $labels);

var_dump($score);
```

```sh
float(0.9484)
```

## Validators
Metrics can be used by themselves or they can be used within a [Validator](https://docs.rubixml.com/en/latest/cross-validation/api.html). Validator objects automate the cross validation process by training and testing a learner on subsets of a provided dataset. The way in which subsets are chosen is based on the validator being used. For example, a [K Fold](https://docs.rubixml.com/en/latest/cross-validation/k-fold.html) validator will automatically select one of k folds of the dataset to use as a validation set and then use the rest to train the learner. It will do this for every unique fold of the dataset such that the model will eventually be tested on every sample.

**Example**

```php
use Rubix\ML\CrossValidation\KFold;
use Rubix\ML\CrossValidation\Metrics\Accuracy;

// Import labeled dataset

$validator = new KFold(10);

$score = $validator->test($estimator, $dataset, new Accuracy());

var_dump($score);
```

```sh
float(0.869)
```

## Reports
Detailed information about the performance of an estimator or specific tasks can be obtained with cross validation [Reports](https://docs.rubixml.com/en/latest/cross-validation/reports/api.html). Reports are more information-dense than metrics, but they can only be used stand-alone. As an example, we can use a [Multiclass Breakdown](https://docs.rubixml.com/en/latest/cross-validation/reports/multiclass-breakdown.html) report to show the performance of a classifier broken down by class label.

```php
use Rubix\ML\CrossValidation\Reports\MulticlassBreakdown;

// Import labels and make predictions

$report = new MulticlassBreakdown();

$result = $report->generate($predictions, $labels);

var_dump($result);
```

```sh
...
['classes']=> array(2) {
	['wolf']=> array(19) {
      	['accuracy']=> float(0.6)
      	['precision']=> float(0.66666666666667)
      	['recall']=> float(0.66666666666667)
      	['specificity']=> float(0.5)
      	['negative_predictive_value']=> float(0.5)
      	['false_discovery_rate']=> float(0.33333333333333)
      	['miss_rate']=> float(0.33333333333333)
      	['fall_out']=> float(0.5)
      	['false_omission_rate']=> float(0.5)
     	['f1_score']=> float(0.66666666666667)
      	['mcc']=> float(0.16666666666667)
      	['informedness']=> float(0.16666666666667)
      	['markedness']=> float(0.16666666666667)
      	['true_positives']=> int(2)
      	['true_negatives']=> int(1)
      	['false_positives']=> int(1)
      	['false_negatives']=> int(1)
      	['cardinality']=> int(3)
      	['density']=> float(0.6)
    }
    ...
```

## Common Problems

### Overfitting
When a model performs well during training but poorly during cross validation it could be that the model has *overfit* the training data. Overfitting occurs when the model conforms too closely to the training data and therefore fails to generalize well to new data or make predictions reliably. Some degree of overfitting can occur with any model, but larger more flexible models are more prone to overfitting due to their ability to *memorize* individual samples.

Overfitting can be controlled and even prevented entirely by smart settings of certain hyper-parameters as most learners employ strategies such as regularization, early stopping, or post-pruning to reduce overfitting. For example, some learners such as [Multi Layer Perceptron](https://docs.rubixml.com/en/latest/classifiers/multi-layer-perceptron.html) and [Gradient Boost](https://docs.rubixml.com/en/latest/regressors/gradient-boost.html) prevent overfitting by determining the point at which the validation score on a holdout portion of the training data starts to decrease.

### Selection Bias
When a model performs well on certain samples but poorly on others it could be that the learner was trained with a dataset that exibits selection bias. Selection bias is the bias introduced when a population is disproportionally represented in a dataset. For example, if a neural network trained to classify pictures of cats and dogs is trained mostly (say 90%) with images of cats, it will likely have difficulty in the real world where cats and dogs are more equally represented.

Although selection bias is largely determined by data collection, there are things you can do within Rubix ML that ensure that unbiased data remains unbiased. For example, the stratified methods on the [Labeled](https://docs.rubixml.com/en/latest/datasets/labeled.html) dataset object split and fold the dataset while maintaining the proportions of labels in each subset.