# Model Ensembles
Ensemble learning is when multiple estimators are used together to make the final prediction of a sample. Model ensembles can consist of multiple variations of the same estimator, a heterogeneous mix of estimators of the same type, or even a mix of different estimator types.

## Bootstrap Aggregator
Bootstrap Aggregation or *bagging* is an ensemble learning technique that trains a set of learners that each specialize on a unique subset of the training set known as a bootstrap set. The final prediction made by the meta-estimator is the averaged prediction returned by the ensemble. In the example below, we'll wrap a [Regression Tree](regressors/regression-tree.md) in a [Bootstrap Aggregator](bootstrap-aggregator.md) meta-estimator to form a *forest* of 1000 trees. By averaging the predictions of the ensemble, we can often achieve greater accuracy than a single estimator at the cost of training more models.

```php
use Rubix\ML\BootstrapAggregator;
use Rubix\ML\Regressors\RegressionTree;

$estimator = new BootstrapAggregator(new RegressionTree(5), 1000);
```

## Committee Machine
[Committee Machine](committee-machine.md) is another meta-estimator ensemble that works by the principal of averaging. It is a voting ensemble consisting of estimators (referred to as *experts*) with user-programmable *influences*. Each expert is trained on the same dataset and the final prediction is based on the contribution of each expert weighted by their influence in the committee. By varying the influences of the experts, we can control which estimators contribute more or less to the final prediction.

```php
use Rubix\ML\CommitteeMachine;
use Rubix\ML\RandomForest;
new Rubix\ML\SoftmaxClassifier;
use Rubix\ML\AdaBoost;

$estimator = new CommitteeMachine([
    new RandomForest(),
    new SoftmaxClassifier(128),
    new AdaBoost(),
], [
    3.0, 1.7, 2.5,
]);
```

## Model Chaining
Model chaining is a form of ensemble learning that uses the predictions of one or more estimators as the input features to other downstream estimators. For example, if our task was to predict if we should give a customer a loan or not, we could first predict the customer's credit score and then add it to the original features for the loan classifier to train and infer on. We can use the [Lambda Function](transformers/lambda-function.md) transformer with a custom callback function to add the new feature to the training set after their values have been predicted. The callback accepts three arguments - the current sample passed by reference, the current row offset in the dataset, and a context variable which we'll use to store the predicted credit scores.

```php
use Rubix\ML\Regressors\KDNeighborsRegressor;
use Rubix\ML\Transformers\LambdaFunction;
use Rubix\ML\Classifiers\RandomForest;
use Rubix\ML\Classifiers\ExtraTreeClassifier;

$creditScoreEstimator = new KDNeighborsRegressor(10);

$creditScoreEstimator->train($dataset);

$creditScores = $creditScoreEstimator->predict($dataset);

$addFeature = function (&$sample, $offset, $context) {
    $sample[] = $context[$offset];
}

$dataset->apply(new LambdaFunction($addFeature, $creditScores));

$loanApprovalEstimator = new RandomForest(new ExtraTreeClassifier(8), 300);

$loanApprovalEstimator->train($dataset);
```

## Model Orchestra
When you combine chaining with model averaging you get a technique referred to as *stacking*. Unlike the [Committee Machine](committee-machine.md) meta-estimator, which relies on a priori knowledge of the estimator influences, stacking aims to learn the influence scores automatically bu using another model. We introduce the orchestra pattern for implementing a stacked model ensemble. The complete model consists of three [Probabilistic](./probabilistic.md) classifiers referred to as the *orchestra* and a *conductor* that makes the final prediction by training on the probabilities outputted by the orchestra. A key step to this process is to separate the training set into two sets so that we can do a second optimization over the second dataset to determine the model influences. We can vary the amount of data used to train each layer of the model by changing the proportion argument to the `stratifiedSplit()` method. For this example, we'll choose to use half of the data to train the orchestra and half to train the conductor.

```php
use Rubix\ML\Classifiers\GaussianNB;
use Rubix\ML\Classifiers\AdaBoost;
use Rubix\ML\Classifiers\KDNeighbors;
use Rubix\ML\Classifiers\MultilayerPerceptron;
use Rubix\ML\NeuralNet\Layers\Dense;
use Rubix\ML\NeuralNet\Layers\Activation;
use Rubix\ML\NeuralNet\ActivationFunctions\ReLU;
use Rubix\ML\Datasets\Labeled;

[$dataset1, $dataset2] = $training->stratifiedSplit(0.5);

$orchestra = [
    new GaussianNB(),
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

$conductor = new MultilayerPerceptron([
    new Dense(100),
    new Activation(new ReLU()),
]);

$conductor->train($dataset);
```
