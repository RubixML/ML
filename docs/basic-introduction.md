# Basic Introduction
In this basic introduction to machine learning in Rubix ML, you'll learn how to structure a project, train a learner to predict successful marriages, and then test the model for accuracy. We assume that you already have a basic understanding of the different types of machine learning such as classification and regression. If not, we recommend the section on [What is Machine Learning?](what-is-machine-learning.md) to start with.

## Obtaining Data
Machine learning projects typically begin with a question. For example, you might want to answer the question of "who of my friends are most likely to stay married to their partner?" One way to go about answering this question with machine learning would be to go out and ask a bunch of happily married and divorced couples the same set of questions about their partner and then use that data to build a model to predict successful relationships based on the answers they gave you. In ML terms, the answers you collect are the values of the *features* that constitute measurements of the phenomena being observed. The number of features in a sample is called the *dimensionality* of the sample. For example, a sample with 10 features is said to be *10-dimensional*.

Suppose that you went out and asked 4 couples (2 married and 2 divorced) to rate their partner's communication skills (between 1 and 5), attractiveness (between 1 and 5), and time spent together per week (hours per week). You would structure the data in PHP like in the example below. You'll notice that the samples are represented in a 2-d array (or *matrix*) and the labels are represented as a 1-d array.

```php
$samples = [
    [3, 4, 50.5], [1, 5, 24.7], [4, 4, 62.0], [3, 2, 31.1]
];

$labels = ['married', 'divorced', 'married', 'divorced'];
```

> **Hint:** See the [Representing your Data](representing-your-data.md) section for an in-depth description of how Rubix ML treats various forms of data.

## The Dataset Object
In Rubix ML, data are passed in specialized containers called [Dataset objects](datasets/api.md). Dataset objects handle selecting, subsampling, transforming, randomizing, and sorting of the samples and labels for you. In general, there are two types of datasets, *Labeled* and *Unlabeled*. Labeled datasets are used for supervised learning and for providing the ground-truth during testing. Unlabeled datasets are used for unsupervised learning and for making predictions (*inference*) on unknown samples.

You could construct a [Labeled](datasets/labeled.md) dataset from the data we collected earlier by passing the samples and labels into the constructor like in the example below.

```php
use Rubix\ML\Datasets\Labeled;

$dataset = new Labeled($samples, $labels);
```

> **Hint:** See the [Extracting Data](extracting-data.md) section to learn more about extracting data from different storage mediums.

## Choosing an Estimator
[Estimators](estimator.md) make up the core of the Rubix ML library. They provide the `predict()` API and are responsible for making predictions on unknown samples. Estimators that can be trained with data are called [Learners](learner.md) and must be trained before making predictions.

In practice, one will experiment with a number of estimators to find the one that works best for their dataset. For our example, we'll focus on an intuitable distance-based supervised learner called K Nearest Neighbors. KNN is a *classifier* because it takes unknown samples and assigns them a class label. In our example, the output of KNN will either be *married* or *divorced* since those are the class labels that we'll train it with.

## Instantiating the Learner
Like most estimators in Rubix ML, the [K Nearest Neighbors](classifiers/k-nearest-neighbors.md) classifier requires a set of parameters (called *hyper-parameters*) to be chosen up-front by the user. These parameters are defined in the class's constructor and control how the learner behaves during training and inference. Hyper-parameters can be selected based on some prior knowledge of the problem space, or completely at random. The defaults provided in Rubix ML are a good place to start for most problems.

The K Nearest Neighbors classifier works by locating the closest training samples to an unknown sample and choosing the class label that is most common. The hyper-parameter *k* is the number of nearest points from the training set to compare an unknown sample to in order to infer its class label. For example, if the 3 closest neighbors to a given unknown sample have 2 married and 1 divorced label, then the algorithm will output a prediction of married since its the most common. To instantiate the KNN learner, pass a set of hyper-parameters to the class's constructor. For this example, let's set *k* to 3 and leave the rest of the hyper-parameters as their default.

```php
use Rubix\ML\Classifiers\KNearestNeighbors;

$estimator = new KNearestNeighbors(3);
```

## Training
Training is the process of feeding the learning algorithm data so that it can build an internal representation (or *model*) of the task its trying to learn. This representation consists of all of the parameters (except hyper-parameters) that are required to make a prediction. In the case of K Nearest Neighbors, this representation is a high-dimensional Euclidean space in which each sample is considered a point. We define the *distance* between two points as their degree of dissimilarity. Thus, points in this space that are similar will be closer together.

To start training, pass the training dataset as a argument to the `train()` method on the learner instance.

```php
$estimator->train($dataset);
```

We can verify that the learner has been trained by calling the `trained()` method.

```php
var_dump($estimator->trained());
```

```sh
bool(true)
```

For our small training set, the entire process should only take a matter of microseconds, but larger datasets with more features can take much longer. Now that the learner is trained, in the next section we'll show how we can feed in unknown samples to generate predictions.

> **Hint:** See the [Training](training.md) section of the docs for a closer look at training a learner.

## Making Predictions
Suppose that we went out and collected 4 new data points from different friends using the same questions we asked the couples we interviewed for our training set. We could predict whether or not they will stay married to their spouse by taking their answers and passing them in an [Unlabeled](datasets/unlabeled.md) dataset to the `predict()` method on the KNN estimator. The process of making predictions is called *inference* because the estimator uses the model constructed during training to infer the label of the unknown samples.

> **Note:** If you attempt to make predictions using an untrained learner, it will throw an exception.

```php
use Rubix\ML\Datasets\Unlabeled;

$samples = [
    [4, 3, 44.2], [2, 2, 16.7], [2, 4, 19.5], [3, 3, 55.0],
];

$dataset = new Unlabeled($samples);

$predictions = $estimator->predict($dataset);

var_dump($predictions);
```

```sh
array(4) {
	[0] => 'married'
	[1] => 'divorced'
	[2] => 'divorced'
	[4] => 'married'
}
```

The output of the estimator are the predicted class labels of the unknown samples in order. We could either trust these predictions as-is or we could proceed to further evaluate the model. In the next section, we'll learn how to test the accuracy of the model using a process called cross validation.

> **Hint:** Check out the section on [Inference](inference.md) for more info on making predictions.

## Model Evaluation
To test that the estimator can correctly generalize what it has learned during training to the real world we can use a process called *cross validation*. The goal of cross validation is to train and test the learner on different subsets of the dataset in order to produce a validation score. For the purposes of the introduction, we will use a simple [Hold Out](cross-validation/hold-out.md) validator which takes a portion of the dataset for testing and leaves the rest for training. The reason we do not use *all* of the data for training is because we want to test the estimator on samples that it has never seen before.

The Hold Out validator requires the user to set the ratio of testing to training samples as a constructor parameter. Let's choose to use a factor of 0.2 (20%) of the dataset for testing leaving the rest (80%) for training.

> **Note:** 20% is a good default choice however your mileage may vary. The important thing to note here is the trade off between more data for training and more data to produce better testing results.

The `test()` method on the validator requires a compatible validation [Metric](https://docs.rubixml.com/en/latest/cross-validation/metrics/api.html) to be selected as the scoring function. To return a score from the Hold Out validator using the [Accuracy](cross-validation/metrics/accuracy.md) metric, pass in an untrained estimator instance along with the entire dataset like in the example below.

```php
use Rubix\ML\CrossValidation\HoldOut;
use Rubix\ML\CrossValidation\Metrics\Accuracy;

$validator = new HoldOut(0.2);

$score = $validator->test($estimator, $dataset, new Accuracy());

var_dump($score);
```

```sh
float(0.945)
```

The output is the validation score which can be interpreted as the degree to which the learner is able to accurately generalize its training to unseen data. According to the example above, our model is about 95% accurate given our chosen metric. Nice work!

> **Hint:** More info can be found in the [Cross Validation](cross-validation.md) section of the docs.

## Next Steps
Congratulations! You've completed the basic introduction to machine learning in PHP with Rubix ML. For a more in-depth tutorial using the K Nearest Neighbors classifier and a real dataset, check out the [Divorce Predictor](https://github.com/RubixML/Divorce) tutorial and example project. Have fun!