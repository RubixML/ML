# Cross Validation
Cross Validation (CV) or *out-of-sample* testing is the primary technique for assessing the generalization performance of a model using data it has never seen before. The validation score gives us a sense for how well the model will perform in the real world. In addition, it allows the user to identify problems such as underfitting, overfitting, and selection bias which are discussed in the last section.

## Metrics
Cross validation [Metrics](cross-validation/metrics/api.md) are used to score the predictions made by an estimator with respect to their ground-truth labels. There are different metrics for different estimator types as shown in the table below. All metrics follow the schema that higher scores are better - thus, common *loss functions* used as metrics such as [Mean Squared Error](https://docs.rubixml.com/en/latest/cross-validation/metrics/mean-squared-error.html) and [RMSE](https://docs.rubixml.com/en/latest/cross-validation/metrics/rmse.html) are given as their *negative* score to conform to this schema.

| Task | Metrics |
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
Metrics can be used stand-alone or they can be used within a [Validator](cross-validation/api.md) as the scoring function. Validators automate cross validation by training and testing a learner on different subsets of a master dataset. The way in which subsets are chosen depends on the algorithm the validator employs under the hood. For example, a [K Fold](cross-validation/k-fold.md) validator will automatically select one of k *folds* of the dataset to use as a validation set and then use the rest of the folds to train the learner. It will do this until the model is trained and tested on every sample in the dataset at least once.

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
Cross validation [Reports](cross-validation/reports/api.md) give you a deeper sense as to how a model is performing. They provide finer-grained details than a single metric but they can only be used stand-alone. As an example, we can use the [Multiclass Breakdown](cross-validation/reports/multiclass-breakdown.md) report to output a number of classification metrics broken down by class label.

| Task | Reports |
|---|---|
| Classification | [Multiclass Breakdown](cross-validation/reports/multiclass-breakdown.md), [Confusion Matrix](cross-validation/reports/confusion-matrix.md) |
| Regression | [Residual Analysis](cross-validation/reports/residual-analysis.md) |
| Clustering | [Contingency Table](cross-validation/reports/contingency-table.md) |
| Anomaly Detection | [Multiclass Breakdown](cross-validation/reports/multiclass-breakdown.md), [Confusion Matrix](cross-validation/reports/confusion-matrix.md) |

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
A poorly performing model can sometimes be explained as *underfiting* the training data - a condition in which the learner is unable to capture the underlying pattern or trend given the model constraints. Underfitting can occur when a simple model, such as the linear one learned by [Logistic Regression](classifiers/logistic-regression.md), is trained on data with complex non-linear processes. In such a case, either model constraints can be relaxed or a new learner with greater flexibility can be selected for the task instead. Adding more features such as by hand engineering or using a transformer like [Polynomial Expander](transformers/polynomial-expander.md) can also help.

### Overfitting
When a model performs well during training but poorly during cross validation it could be that the model has *overfit* the training data. Overfitting occurs when the model conforms too closely to the training data and therefore fails to generalize well to new data or make predictions reliably. Some degree of overfitting can occur with any model, but larger more flexible models are more prone to overfitting due to their ability to *memorize* individual samples.

Most learners in Rubix ML employ strategies such as regularization, early stopping, or post-pruning to reduce overfitting. For example, some learners such as [Multilayer Perceptron](classifiers/multilayer-perceptron.md) and [Gradient Boost](regressors/gradient-boost.md) prevent overfitting by stopping at the point at which the validation score on a holdout set starts to decrease. Adding more samples to the dataset can also help to reduce overfitting.

### Selection Bias
When a model performs well on certain samples but poorly on others it could be that the learner was trained with a dataset that exhibits selection bias. Selection bias is the bias introduced when a population is disproportionally represented in a dataset. For example, if a neural network trained to classify pictures of cats and dogs is trained mostly (say 90%) with images of cats, it will likely have difficulty in the real world where cats and dogs are more equally represented.

Although selection bias is largely determined through data collection, there are a number of things that you can do within Rubix ML to ensure that unbiased data remains unbiased. For example, the *stratified* methods on the [Labeled](datasets/labeled.md) dataset object split and fold the dataset while maintaining the proportions of labels in each subset.