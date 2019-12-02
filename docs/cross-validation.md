# Cross Validation
Cross Validation (CV) or *out-of-sample* testing is the primary technique for assessing the generalization performance of a model using data it has never seen before. The validation score gives us a sense for how well the model will perform in the real world. In addition, it allows the user to identify problems such as underfitting, overfitting, or selection bias which are discussed in the last section.

## Metrics
Cross validation [Metrics](cross-validation/metrics/api.md) are used to score the predictions made by an estimator with respect to their ground-truth labels. There are different metrics for different estimator types. For example, one can measure the accuracy of a classifier or the mean squared error (MSE) of a regressor.

| Task | Example Metrics |
|---|---|
| Classification | [Accuracy](cross-validation/metrics/accuracy.md), [F Beta](cross-validation/metrics/f-beta.md), [MCC](cross-validation/metrics/mcc.md), [Informedness](cross-validation/metrics/informedness.md) |
| Regression | [Mean Absolute Error](cross-validation/metrics/mean-absolute-error.md), [R Squared](cross-validation/metrics/r-squared.md), [SMAPE](cross-validation/metrics/smape.md) |
| Clustering | [Homogeneity](cross-validation/metrics/homogeneity.md), [V Measure](cross-validation/metrics/v-measure.md), [Rand Index](cross-validation/metrics/rand-index.md) |
| Anomaly Detection | [F Beta](cross-validation/metrics/f-beta.md), [MCC](cross-validation/metrics/mcc.md) |

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
Metrics can be used by themselves or they can be used within a [Validator](cross-validation/api.md). Validator objects automate the cross validation process by training and testing a learner on subsets of a provided dataset. The way in which subsets are chosen is based on the algorithm that the validator is employing. For example, a [K Fold](cross-validation/k-fold.md) validator will automatically select one of k folds of the dataset to use as a validation set and then use the rest to train the learner. It will do this for every unique fold such that the model is eventually tested on every sample in the dataset.

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
Detailed information about the performance of an estimator or specific tasks can be obtained with cross validation [Reports](cross-validation/reports/api.md). Reports are more information-dense than metrics, but they can only be used stand-alone. As an example, we can use a [Multiclass Breakdown](cross-validation/reports/multiclass-breakdown.md) report to show the performance of a classifier broken down by class label.

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
Here are some common problems that cross validation can help identify.

### Underfitting
A poorly performing model can sometimes be explained as *underfiting* the training data - a condition in which the learner is unable to capture the underlying pattern or trend given the model constraints. Underfitting can occur when a simple model, such as the linear one learned by [Logistic Regression](classifiers/logistic-regression.md), is trained on data with complex non-linear processes. In such a case, either model constraints can be relaxed or a new learner with greater flexibility such as a [Random Forest](classifiers/random-forest.md) can be selected for the task instead. Adding more features such as by hand engineering or automatically using a transformer like [Polynomial Expander](transformers/polynomial-expander.md) can also help.

### Overfitting
When a model performs well during training but poorly during cross validation it could be that the model has *overfit* the training data. Overfitting occurs when the model conforms too closely to the training data and therefore fails to generalize well to new data or make predictions reliably. Some degree of overfitting can occur with any model, but larger more flexible models are more prone to overfitting due to their ability to *memorize* individual samples.

Most learners employ strategies such as regularization, early stopping, or post-pruning to reduce overfitting. In such case, overfitting can be controlled and prevented entirely. For example, some learners such as [Multilayer Perceptron](classifiers/multilayer-perceptron.md) and [Gradient Boost](regressors/gradient-boost.md) prevent overfitting by determining the point at which the validation score on a holdout portion of the training data starts to decrease. Adding more samples to your dataset can also help to reduce overfitting.

### Selection Bias
When a model performs well on certain samples but poorly on others it could be that the learner was trained with a dataset that exibits selection bias. Selection bias is the bias introduced when a population is disproportionally represented in a dataset. For example, if a neural network trained to classify pictures of cats and dogs is trained mostly (say 90%) with images of cats, it will likely have difficulty in the real world where cats and dogs are more equally represented.

Although selection bias is largely determined through data collection, there are a number of things that you can do within Rubix ML to ensure that unbiased data remains unbiased. For example, the stratified methods on the [Labeled](datasets/labeled.md) dataset object split and fold the dataset while maintaining the proportions of labels in each subset.