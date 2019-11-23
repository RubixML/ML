# Basic Introduction
In this basic introduction to machine learning in Rubix ML, you'll learn how to structure a project, train a learner to predict successful marriages, and then test the model for accuracy. We assume that you already have a basic understanding of the different types of machine learning such as classification and regression. If not, we recommend the section on [What is Machine Learning?](what-is-machine-learning.md) to start with.

# Obtaining Data
Machine learning projects typically begin with a question. For example, you might want to answer the question of "who of my friends are most likely to stay married to their partner?" One way to go about answering this question with machine learning would be to go out and ask a bunch of happily married and divorced couples the same set of questions about their partner and then use that data to build a model to predict successful relationships based on the answers they gave you. In ML terms, the answers you collect are the values of the *features* that constitute measurements of the phenomena being observed. The number of features in a sample is called the *dimensionality* of the sample. For example, a sample with 20 features is said to be *20 dimensional*.

As an alternative to collecting data yourself, you can access one of the many open datasets that are free to use from a public repository. The advantages of using a public dataset is that the data has most likely already been cleaned and prepared for you. We recommend the University of California Irvine [Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets.php) as a great place to get started with using open datasets.

> **Hint:** See the ['Extracting Data'](extracting-data.md) section to learn more about extracting data from different storage formats.

## The Dataset Object
In Rubix ML, data are passed in specialized containers called [Dataset objects](datasets/api.md). Dataset objects handle selecting, subsampling, transforming, randomizing, and sorting of the samples and labels for you. In general, there are two types of datasets, *Labeled* and *Unlabeled*. Labeled datasets are used for supervised learning and for providing the ground-truth during testing. Unlabeled datasets are used for unsupervised learning and for making predictions (*inference*) on unknown samples.

Suppose that you went out and asked 4 couples (2 married and 2 divorced) to rate their partner's communication skills (between 1 and 5), attractiveness (between 1 and 5), and time spent together per week (hours per week). You could construct a [Labeled](datasets/labeled.md) dataset from this data by passing the samples and labels into the constructor like in the example below.

```php
use Rubix\ML\Datasets\Labeled;

$samples = [
    [3, 4, 50.5], [1, 5, 24.7], [4, 4, 62.0], [3, 2, 31.1]
];

$labels = ['married', 'divorced', 'married', 'divorced'];

$dataset = new Labeled($samples, $labels);
```

> **Hint:** See the ['Representing your Data'](representing-your-data.md) section for an in-depth description of how Rubix ML treats various forms of data.

# Choosing an Estimator
[Estimators](https://docs.rubixml.com/en/latest/estimator.html) make up the core of the Rubix ML library. They provide the `predict()` API and are responsible for making predictions on unknown samples. Estimators that can be trained with data are called [Learners](https://docs.rubixml.com/en/latest/learner.html) and must be trained before making predictions.

For our example we will focus on an intuitable distance-based supervised learner called [K Nearest Neighbors](classifiers/k-nearest-neighbors.md). KNN is a type of estimator called a Classifier because it takes unknown samples and assigns them a class label. In our example the output of KNN will either be *married* or *divorced* since those are the class labels that we train it with.

## Creating the Estimator Instance
The K Nearest Neighbors classifier works by locating the closest training samples to an unknown sample and choosing the class label that is most common. Like most estimators, the K Nearest Neighbors (KNN) classifier requires a set of parameters (called *hyper-parameters*) to be chosen up-front by the user. These parameters control how the learner behaves during training and inference. These parameters can be selected based on some prior knowledge of the problem space, or completely at random. The defaults provided in Rubix ML are a good place to start for most problems.

In KNN, the hyper-parameter *k* is the number of nearest points from the training set to compare an unknown sample to in order to infer its class label. For example, if the 5 closest neighbors to a given unknown sample have 4 married and 1 divorced label, then the algorithm will output a prediction of married with a probability of 0.8.

```php
use Rubix\ML\Classifiers\KNearestNeighbors;

$estimator = new KNearestNeighbors(5);
```

# Training the Learner
Training is the process of feeding the learning algorithm data so that it can build an internal representation of the problem space. This representation is often called a *model* and it consists of all of the parameters (except hyper-parameters) that are required to make a prediction. In the case of K Nearest Neighbors, this representation is a high-dimensional Euclidean space in which each sample is considered a point.

> **Note:** If you try to make predictions using an untrained learner, it will throw an exception.

```php
$estimator->train($dataset);
```

We can verify that the learner has been trained by calling the `trained()` method:
```php
var_dump($estimator->trained());
```

```sh
bool(true)
```

For our small training set, the training process should only take a matter of microseconds, but larger datasets with higher dimensionality can take much longer. Once the learner has been trained, we can feed in some unknown samples to see what the model predicts.

> **Hint:** See the ['Training'](training.md) section for a closer look at training a learner.

### Making Predictions
Suppose that we went out and collected 4 new data points from our friends using the same questions we asked the couples we interviewed for our training set. We could predict whether or not they will stay married by taking their answers and running them through the trained KNN estimator in and [Unlabeled](https://docs.rubixml.com/en/latest/datasets/unlabeled.html) dataset. The process of making predictions is called *inference* because the estimator uses the model constructed during training to infer the label of the unknown samples.

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

The output of the KNN classifier are the predicted class labels of the unknown samples in the order they were feed to the estimator. We could either trust these predictions or we could procees to further evaluate the model. In the next section, we'll learn how to test the generalization performance of our estimator.

> **Hint:** Check out the section on ['Inference'](inference.md) for more info on making predictions with an estimator.

# Model Evaluation
To test that the estimator can correctly generalize what it has learned during training to the real world we use a process called *cross validation*. The goal of cross validation is to train and test the learner on different subsets of the dataset in  order to produce a validation score. For the purposes of the introduction, we will use the Hold Out validator which takes a portion of the dataset for testing and leaves the rest for training. The reason we do not use *all* of the data for training is because we want to test the estimator on samples that it has never seen before.

The [Hold Out](cross-validation/hold-out.md) validator requires the user to set the ratio of testing to training samples as a constructor parameter. Let's choose to use a factor of 0.2 (20%) of the dataset for testing leaving the rest (80%) for training.

> **Note:** Typically, 0.2 is a good default choice however your mileage may vary. The important thing to note here is the trade off between more data for training and more data to produce better testing results.

To return a score from the Hold Out validator using the [Accuracy](https://docs.rubixml.com/en/latest/cross-validation/metrics/accuracy.html) metric, pass in an untrained estimator instance along with the entire dataset.

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

The output of the cross validator is a validation score that can be interpretted as the degree to which the learner is able to accurately generalize its training to unknown data. In the example above, our model is about 95% accurate according to our chosen metric.

> **Hint:** More info can be found in the ['Cross Validation'](cross-validation.md) section of the docs.

### Next Steps
Congratulations! You've completed the basic introduction to machine learning in PHP with Rubix ML. For a more in-depth tutorial using the K Nearest Neighbors classifier, check out the [Iris Flower](https://github.com/RubixML/Iris) example project. We highly recommend browsing the rest of the documentation and the other [example projects](https://github.com/RubixML) which range from beginner to advanced skill level. Have fun and stay curious!