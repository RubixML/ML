# What is Machine Learning?
Machine learning (ML) is the process by which a computer program is able to progressively improve performance on a task through *training* and data without explicitly being programmed. It is a way of programming with data. After training, we can use the estimator to make predictions about future outcomes (referred to as *inference*). There are two types of machine learning that Rubix supports out of the box - Supervised and Unsupervised.

## Supervised Learning
Supervised learning is a type of machine learning that includes a training signal in the form of human annotations called *labels* along with the training samples. The training signal guides the learner to output predictions that resemble values found in the labels of the training set.

### Classification
For classification problems, a learner is trained to differentiate samples among a set of K possible discrete classes. In this type of problem, the labels provided are the classes that each sample belongs to such as *"cat"*, *"dog"*, or *"ship"*. Examples of classification problems include image recognition, text sentiment analysis, and Iris flower species classification.

> **Note:** As a convention throughout Rubix ML, discrete variables are always denoted by string type.

### Regression
Unlike classification, where the number of possible class predictions an estimator can make is 1 of K, a Regressor can predict an infinite range of continuous values. In this case, the labels are the desired output values of the estimator given each training sample as input. Regression problems include determining the angle of an automobile steering wheel, estimating the sale price of a home, and credit scoring.

## Unsupervised Learning
A form of learning that does not require training labels is referred to as Unsupervised learning. Instead, Unsupervised learners aim to detect patterns in raw data. Since it is not always easy or possible to obtain labeled data, an Unsupervised learning method is often the first step in discovering knowledge about your data.

### Clustering
Clustering takes a dataset of unlabeled samples and assigns each a discrete label based on their similarity to each other. Example of where clustering is used is in tissue differentiation in PET scan images, customer database market segmentation, or to discover communities within social networks.

### Anomaly Detection
Anomaly detection is the flagging and/or ranking of samples within a dataset based on how different or rare they are. Anomaly detection is commoonly used in security for intrusion and denial of service detection, and in the financial industry to detect fraud.

### Manifold Learning
Manifold learning is a type of non-linear dimensionality reduction used for densely embedding feature representations such as for visualizing high dimensional datasets in low (1 to 3) dimensions and modelling language as word vectors.

# Obtaining Data
Machine learning projects typically begin with a question. For example, you might want to answer the question "who of my friends are most likely to stay married to their partner?" One way to go about answering this question with machine learning would be to go out and ask a bunch of happily married and divorced couples the same set of questions about their partner and then use that data to build a model to predict successful or doomed relationships based on the answers your friends give you. In ML terms, the answers you collect are called *features* and they constitute measurements of some phenomena being observed. The number of features in a sample is called the *dimensionality* of the sample. For example, a sample with 20 features is said to be *20 dimensional*.

An alternative to collecting data yourself is to download one of the many open datasets that are free to use from a public repository. The advantages of using a public dataset is that, usually, the data has already been cleaned and prepared for you. We recommend the University of California Irvine [Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets.php) as a great place to get started with using open source datasets.

## Extracting Data
Before data can become useful, we need to load it into the computer in the proper format. This involves extracting the data from source before anything else. There are many PHP libraries that help make extracting data from various sources easy and intuitive, and we recommend checking them out as a great place to start.

- [PHP League CSV](https://csv.thephpleague.com/) - Generator-based CSV extractor
- [Doctrine DBAL](https://www.doctrine-project.org/projects/dbal.html) - SQL database abstraction layer
- [Google BigQuery](https://cloud.google.com/bigquery/docs/reference/libraries) - Cloud-based data warehouse via SQL

## The Dataset Object
In Rubix, data is passed around in specialized data containers called Dataset objects. [Datasets](datasets/api.md) internally handle selecting, splitting, folding, transforming, and randomizing the samples and labels contained within. In general, there are two types of datasets, *Labeled* and *Unlabeled*. Labeled datasets are used for supervised learning and for providing the ground-truth during cross validation. Unlabeled datasets are used for unsupervised learning and for making predictions (*inference*) on unknown samples.

As a simplistic example, suppose that you went out and asked 100 couples (50 married and 50 divorced) to rate their partner's communication skills (between 1 and 5), attractiveness (between 1 and 5), and time spent together per week (hours per week). You could construct a [Labeled](datasets/labeled.md) Dataset object from this data by passing the samples and labels into the constructor.

```php
use Rubix\ML\Datasets\Labeled;

$samples = [
    [3, 4, 50.5], [1, 5, 24.7], [4, 4, 62.0], [3, 2, 31.1]
];

$labels = ['married', 'divorced', 'married', 'divorced'];

$dataset = new Labeled($samples, $labels);
```

# Choosing an Estimator
Estimators make up the core of the Rubix library as they are responsible for making predictions. There are many different estimators to choose from and each one operates differently. Choosing the right [Estimator](estimator.md) for the job is crucial to creating a performant system.

For our simple example we will focus on an easily intuitable classifier called [K Nearest Neighbors](classifiers/k-nearest-neighbors.md). Since the label of each training sample we collect will be a discrete class (*married couples* or *divorced couples*), we need an Estimator that is designed to output class predictions. The K Nearest Neighbors classifier works by locating the closest training samples to an unknown sample and choosing the class label that appears most often.

> **Note:** In practice, you will test out a number of different estimators to get the best sense of what works for your particular dataset.

## Creating the Estimator Instance
Like most estimators, the K Nearest Neighbors (KNN) classifier requires a set of parameters (called *hyper-parameters*) to be chosen up front by the user. These parameters control how the learner behaves during training and inference. These parameters can be selected based on some prior knowledge of the problem space, or at random. The defaults provided in Rubix are a good place to start for most machine learning problems.

In K Nearest Neighbors, the hyper-parameter *k* is the number of nearest points from the training set to compare an unknown sample to in order to infer its class label. For example, if the 5 closest neighbors to a given unknown sample have 4 married labels and 1 divorced label, then the algorithm will output a prediction of married with a probability of 0.8.

The second hyper-parameter is the distance *kernel* that determines how distance is measured within the model. We'll go with standard [Euclidean](kernels/distance/euclidean.md) distance for now.

Then, to instantiate the K Nearest Neighbors classifier ...

```php
use Rubix\ML\Classifiers\KNearestNeighbors;
use Rubix\ML\Kernels\Distance\Euclidean;

$estimator = new KNearestNeighbors(5, new Euclidean());
```

# Training and Prediction
Training is the process of feeding the learning algorithm data so that it can build a model of the problem. A trained model consists of all of the parameters (except hyper-parameters) that are required for the estimator to make predictions. If you try to make predictions using an untrained learner, it will throw an exception.

Passing the Labeled dataset to the instantiated learner, we can train our K Nearest Neighbors classifier like so:
```php
$estimator->train($dataset);
```

We can verify that the learner has been trained by calling the `trained()` method:
```php
var_dump($estimator->trained());
```

**Output**

```sh
bool(true)
```

For our 100 sample example training set, training should only take a matter of microseconds, but larger datasets with higher dimensionality and fancier learning algorithms can take much longer. Once the estimator has been fully trained, we can now feed in some unknown samples to see what the model predicts.

Turning back to our example problem, suppose that we went out and collected 5 new data points from our friends using the same questions we asked the couples we interviewed for our training set. We could make predictions on whether they will stay married or get divorced by taking their answers as features and running them in an Unlabeled dataset through the trained Estimator's `predict()` method.
```php
use Rubix\ML\Datasets\Unlabeled;

$unknown = [
    [4, 3, 44.2], [2, 2, 16.7], [2, 4, 19.5], [1, 5, 8.6], [3, 3, 55.0],
];

$dataset = new Unlabeled($unknown);

$predictions = $estimator->predict($dataset);

var_dump($predictions);
```

**Output**

```sh
array(5) {
	[0] => 'married'
	[1] => 'divorced'
	[2] => 'divorced'
	[3] => 'divorced'
	[4] => 'married'
}
```

# Model Evaluation
Making predictions is not very useful unless the estimator can correctly generalize what it has learned during training to the real world. Cross Validation is a process by which we can test the model for its generalization ability. For the purposes of this introduction, we will use a simple form of cross validation called *Hold Out*. The [Hold Out](cross-validation/hold-out.md) validator will take care of splitting the dataset into training and testing sets automatically, such that a portion of the data is *held out* to be used for testing (or *validating*) the model. The reason we do not use *all* of the data for training is because we want to test the Estimator on samples that it has never seen before.

The Hold Out validator requires you to set the ratio of testing to training samples as a constructor parameter. In this case, let's choose to use a factor of 0.2 (20%) of the dataset for testing leaving the rest (80%) for training. Typically, 0.2 is a good default choice however your mileage may vary. The important thing to note here is the trade off between more data for training and more data to produce precise testing results.

To return a score from the Hold Out validator using the Accuracy metric just pass it the untrained estimator instance and a dataset:

```php
use Rubix\ML\CrossValidation\HoldOut;
use Rubix\ML\CrossValidation\Metrics\Accuracy;

$validator = new HoldOut(0.2);

$score = $validator->test($estimator, $dataset, new Accuracy());

var_dump($score);
```

**Output**

```sh
float(0.945)
```

# Next Steps
Congratulations, you're done with the basic introduction to machine learning in Rubix ML. For a more in-depth tutorial using the K Nearest Neighbors classifier, check out the [Iris Flower](https://github.com/RubixML/Iris) example project. From here, we highly recommend browsing the rest of the documentation and the other [example projects](https://github.com/RubixML) which range from beginner to advanced skill level.