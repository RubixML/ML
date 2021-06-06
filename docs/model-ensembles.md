# Model Ensembles
Ensemble learning is when multiple estimators are used together to make the final prediction of a sample. Meta-estimator ensembles can consist of multiple variations of the same estimator or a heterogeneous mix of estimators of the same type. They generally work by the principal of averaging and can often achieve greater accuracy than a single estimator at the cost of training more models.

## Bootstrap Aggregator
Bootstrap Aggregation or *bagging* is an ensemble learning technique that trains a set of learners that each specialize on a unique subset of the training set known as a bootstrap set. The final prediction made by the meta-estimator is the averaged prediction returned by the ensemble. In the example below, we'll wrap a [Regression Tree](regressors/regression-tree.md) in a [Bootstrap Aggregator](bootstrap-aggregator.md) to form a *forest* of 1000 trees.

```php
use Rubix\ML\BootstrapAggregator;
use Rubix\ML\Regressors\RegressionTree;

$estimator = new BootstrapAggregator(new RegressionTree(5), 1000);
```

## Committee Machine
[Committee Machine](committee-machine.md) is a voting ensemble consisting of estimators (referred to as *experts*) with user-programmable *influences*. Each expert is trained on the same dataset and the final prediction is based on the contribution of each expert weighted by their influence in the committee.

```php
use Rubix\ML\CommitteeMachine;
use Rubix\ML\RandomForest;
new Rubix\ML\SoftmaxClassifier;
use Rubix\ML\AdaBoost;
use Rubix\ML\ClassificationTree;
use Rubix\ML\Backends\Amp;

$estimator = new CommitteeMachine([
    new RandomForest(300),
    new SoftmaxClassifier(128),
    new AdaBoost(new ClassificationTree(5), 1.0),
], [
    3.0, 1.0, 2.0, // Influence scores
]);
```

## Cascading ML
Another type of ensemble learning is a form that uses the predictions of one or more estimators as input features to other downstream estimators. For example, if our task was to predict if we should give a customer a loan or not, we could first predict the customer's credit score and then feed the score as a new feature to our loan classifier. We can use the [Lambda Function](transformers/lambda-function.md) transformer and a custom callback to add the new feature to the training set after their values have been predicted. The callback accepts three arguments - the current sample passed by reference, the current row offset in the dataset, and a context variable containing the credit scores.

```php
use Rubix\ML\Regressors\KDNeighborsRegressor;
use Rubix\ML\Classifiers\RandomForest;
use Rubix\ML\Transformers\LambdaFunction;

$creditScoreEstimator = new KDNeighborsRegressor(5);

$creditScoreEstimator->train($dataset);

$creditScores = $creditScoreEstimator->predict($dataset);

$addFeature = function (&$sample, $offset, $context) {
    $sample[] = $context[$offset];
}

$dataset->apply(new LambdaFunction($addFeature, $creditScores));

$loanApprovalEstimator = new RandomForest(300);

$loanApprovalEstimator->train($dataset);
```

## Model Orchestra
When you combine cascading with committee voting you get a technique referred to as *stacking*. Unlike the [Committee Machine](committee-machine.md) meta-estimator, which relies on a priori knowledge of the estimator influences, stacking aims to learn the influence scores automatically bu using another model. We introduce the orchestra pattern for implementing a stacked model ensemble. The complete model consists of three [Probabilistic](./probabilistic.md) classifiers referred to as the *orchestra* and a *conductor* that makes the final prediction by training on the probabilities outputted by the orchestra. A key step to this process is to separate the training set into two sets so that we can do a second optimization over the second dataset to determine the model influences. We can vary the amount of data used to train each layer of the model by changing the proportion argument to the `stratifiedSplit()` method. For this example, we'll choose to use half of the data to train the orchestra and half to train the conductor.

```php
use Rubix\ML\Classifiers\LogisticRegression;
use Rubix\ML\Classifiers\AdaBoost;
use Rubix\ML\Classifiers\KDNeighbors;
use Rubix\ML\Classifiers\MultilayerPerceptron;
use Rubix\ML\NeuralNet\Optimizers\Stochastic;
use Rubix\ML\Datasets\Labeled;

[$dataset1, $dataset2] = $training->stratifiedSplit(0.5);

$orchestra = [
    new LogisticRegression(128),
    new AdaBoost(),
    new KDNeighbors(10),
];

$samples = [];

foreach ($orchestra as $estimator) {
    $estimator->train($dataset1);

    $probabilities = $estimator->proba($dataset2);

    foreach ($probabilities as $offset => $dist) {
        $sample = &$sample[$offset];

        $sample = array_merge($sample, array_values($dist));
    }
}

$dataset = new Labeled($samples, $dataset2->labels());

$conductor = new MultilayerPerceptron(128, new Stochastic(0.0001));

$conductor->train($dataset);
```
