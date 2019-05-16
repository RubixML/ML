<a href="https://rubixml.com" target="_blank"><img src="https://raw.githubusercontent.com/RubixML/RubixML/master/docs/images/rubix-ml-logo.svg?sanitize=true" width="250" alt="Rubix ML for PHP" /></a>

[![PHP from Packagist](https://img.shields.io/packagist/php-v/rubix/ml.svg?style=flat-square&colorB=8892BF)](https://www.php.net/) [![Latest Stable Version](https://img.shields.io/packagist/v/rubix/ml.svg?style=flat-square&colorB=orange)](https://packagist.org/packages/rubix/ml) [![Downloads from Packagist](https://img.shields.io/packagist/dt/rubix/ml.svg?style=flat-square&colorB=red)](https://packagist.org/packages/rubix/ml) [![Travis](https://img.shields.io/travis/RubixML/RubixML.svg?style=flat-square)](https://travis-ci.org/RubixML/RubixML) [![GitHub license](https://img.shields.io/github/license/andrewdalpino/Rubix.svg?style=flat-square)](https://github.com/andrewdalpino/Rubix/blob/master/LICENSE.md)

A high-level machine learning library that allows you to build programs that learn from data using the [PHP](https://php.net) language.

- **Easy** and fast prototyping with user-friendly API
- **40+** modern *supervised* and *unsupervised* learners
- **Modular** architecture combines power and flexibility
- **Open source** and free to use commercially

## Installation
Install Rubix ML with [Composer](https://getcomposer.org/):
```sh
$ composer require rubix/ml
```

## Requirements
- [PHP](https://php.net/manual/en/install.php) 7.1.3 or above

#### Optional

- [SVM extension](https://php.net/manual/en/book.svm.php) for Support Vector Machine engine (libsvm)
- [GD extension](https://php.net/manual/en/book.image.php) for image manipulation
- [Redis extension](https://github.com/phpredis/phpredis) for persisting to a Redis DB
- [Igbinary extension](https://github.com/igbinary/igbinary) for fast binary serialization of persistables

## Documentation

### Table of Contents

- [Basic Introduction](#basic-introduction)
	- [Obtaining Data](#obtaining-data)
	- [Choosing an Estimator](#choosing-an-estimator)
	- [Training and Prediction](#training-and-prediction)
	- [Evaluation](#evaluating-model-performance)
	- [Visualization](#visualization)
    - [Next Steps](#next-steps)
- [System Architecture](#system-architecture)
- [Tutorials & Examples](https://github.com/RubixML)
    - [CIFAR-10 Image Recognizer](https://github.com/RubixML/CIFAR-10)
	- [Color Blob Clusterer](https://github.com/RubixML/Colors)
	- [Credit Card Default Predictor](https://github.com/RubixML/Credit)
    - [MNIST Handwritten Digit Recognizer](https://github.com/RubixML/MNIST)
	- [Human Activity Recognizer](https://github.com/RubixML/HAR)
	- [Housing Price Predictor](https://github.com/RubixML/Housing)
	- [Iris Flower Classifier](https://github.com/RubixML/Iris)
	- [Text Sentiment Analyzer](https://github.com/RubixML/Sentiment)
- [API Reference](#api-reference)
    - [Interfaces](#interfaces)
	    - [Estimator](#estimator)
        - [Learner](#learner)
        - [Online](#online)
        - [Parallel](#parallel)
        - [Persistable](#persistable)
        - [Probabilistic](#probabilistic)
        - [Verbose](#verbose)
        - [Wrapper](#wrapper)
	- [Dataset Objects](#dataset-objects)
		- [Labeled](#labeled)
		- [Unlabeled](#unlabeled)
        - [Generators](#generators)
            - [Agglomerate](#agglomerate)
            - [Blob](#blob)
            - [Circle](#circle)
            - [Half Moon](#half-moon)
            - [Swiss Roll](#swiss-roll)
    - [Anomaly Detectors](#anomaly-detectors)
        - [Isolation Forest](#isolation-forest)
        - [K-d LOF](#k-d-lof)
        - [LODA](#loda)
        - [Local Outlier Factor](#local-outlier-factor)
        - [One Class SVM](#one-class-svm)
        - [Robust Z Score](#robust-z-score)
    - [Classifiers](#classifiers)
        - [AdaBoost](#adaboost)
        - [Classification Tree](#classification-tree)
        - [Dummy Classifier](#dummy-classifier)
        - [Extra Tree Classifier](#extra-tree-classifier)
        - [Gaussian Naive Bayes](#gaussian-naive-bayes)
        - [K-d Neighbors](#k-d-neighbors)
        - [K Nearest Neighbors](#k-nearest-neighbors)
        - [Logistic Regression](#logistic-regression)
        - [Multi Layer Perceptron](#multi-layer-perceptron)
        - [Naive Bayes](#naive-bayes)
        - [Radius Neighbors](#radius-neighbors)
        - [Random Forest](#random-forest)
        - [Softmax Classifier](#softmax-classifier)
        - [SVC](#svc)
    - [Clusterers](#clusterers)
        - [DBSCAN](#dbscan)
        - [Fuzzy C Means](#fuzzy-c-means)
        - [Gaussian Mixture](#gaussian-mixture)
        - [K Means](#k-means)
        - [Mean Shift](#mean-shift)
        - [Seeders](#seeders)
            - [K-MC2](#k-mc2)
            - [Plus Plus](#plus-plus)
            - [Random](#random)
    - [Embedders](#embedders)
        - [t-SNE](#t-sne)
    - [Regressors](#regressors)
        - [Adaline](#adaline)
        - [Dummy Regressor](#dummy-regressor)
        - [Extra Tree Regressor](#extra-tree-regressor)
        - [Gradient Boost](#gradient-boost)
        - [K-d Neighbors Regressor](#k-d-neighbors-regressor)
        - [KNN Regressor](#knn-regressor)
        - [MLP Regressor](#mlp-regressor)
        - [Radius Neighbors Regressor](#radius-neighbors-regressor)
        - [Regression Tree](#regression-tree)
        - [Ridge](#ridge)
        - [SVR](#svr)
    - [Meta-Estimators](#meta-estimators)
        - [Bootstrap Aggregator](#bootstrap-aggregator)
        - [Committee Machine](#committee-machine)
        - [Grid Search](#grid-search)
        - [Persistent Model](#persistent-model)
        - [Pipeline](#pipeline)
    - [Backends](#backends)
        - [Amp](#amp)
        - [Serial](#serial)
    - [Persisters](#persisters)
        - [Filesystem](#filesystem)
        - [Redis DB](#redis-db)
        - [Serializers](#serializers)
            - [Binary](#binary-serializer)
            - [Native](#native)
	- [Transformers](#transformers)
		- [Dense Random Projector](#dense-random-projector)
		- [Gaussian Random Projector](#gaussian-random-projector)
		- [HTML Stripper](#html-stripper)
		- [Image Resizer](#image-resizer)
		- [Image Vectorizer](#image-vectorizer)
		- [Interval Discretizer](#interval-discretizer)
		- [L1 Normalizer](#l1-normalizer)
		- [L2 Normalizer](#l2-normalizer)
		- [Lambda Function](#lambda-function)
		- [Linear Discriminant Analysis](#linear-discriminant-analysis)
		- [Max Absolute Scaler](#max-absolute-scaler)
		- [Min Max Normalizer](#min-max-normalizer)
		- [Missing Data Imputer](#missing-data-imputer)
		- [Numeric String Converter](#numeric-string-converter)
		- [One Hot Encoder](#one-hot-encoder)
		- [Polynomial Expander](#polynomial-expander)
		- [Principal Component Analysis](#principal-component-analysis)
		- [Robust Standardizer](#robust-standardizer)
		- [Sparse Random Projector](#sparse-random-projector)
		- [Stop Word Filter](#stop-word-filter)
		- [Text Normalizer](#text-normalizer)
		- [TF-IDF Transformer](#tf-idf-transformer)
		- [Variance Threshold Filter](#variance-threshold-filter)
		- [Word Count Vectorizer](#word-count-vectorizer)
		- [Z Scale Standardizer](#z-scale-standardizer)
	- [Neural Network](#neural-network)
		- [Activation Functions](#activation-functions)
			- [ELU](#elu)
			- [Hyperbolic Tangent](#hyperbolic-tangent)
			- [Leaky ReLU](#leaky-relu)
			- [ReLU](#relu)
			- [SELU](#selu)
			- [Sigmoid](#sigmoid)
			- [Softmax](#softmax)
			- [Soft Plus](#soft-plus)
			- [Softsign](#softsign)
			- [Thresholded ReLU](#thresholded-relu)
		- [Cost Functions](#cost-functions)
			- [Cross Entropy](#cross-entropy)
			- [Huber Loss](#huber-loss)
			- [Least Squares](#least-squares)
			- [Relative Entropy](#relative-entropy)
	    - [Initializers](#initializers)
            - [Constant](#constant)
			- [He](#he)
			- [Le Cun](#le-cun)
		    - [Normal](#normal)
			- [Uniform](#uniform)
		    - [Xavier 1](#xavier-1)
		    - [Xavier 2](#xavier-2)
		- [Layers](#layers)
			- [Input Layers](#input-layers)
				- [Placeholder 1D](#placeholder-1d)
			- [Hidden Layers](#hidden-layers)
		        - [Activation](#activation)
				- [Alpha Dropout](#alpha-dropout)
				- [Batch Norm](#batch-norm)
				- [Dense](#dense)
				- [Dropout](#dropout)
				- [Noise](#noise)
		        - [PReLU](#prelu)
			- [Output Layers](#output-layers)
				- [Binary](#Binary)
				- [Continuous](#continuous)
				- [Multiclass](#multiclass)
		- [Optimizers](#optimizers)
			- [AdaGrad](#adagrad)
			- [Adam](#adam)
            - [AdaMax](#adamax)
			- [Cyclical](#cyclical)
			- [Momentum](#momentum)
			- [RMS Prop](#rms-prop)
			- [Step Decay](#step-decay)
			- [Stochastic](#stochastic)
	- [Kernels](#kernels)
		- [Distance](#distance)
			- [Canberra](#canberra)
			- [Cosine](#cosine)
			- [Diagonal](#diagonal)
			- [Euclidean](#euclidean)
			- [Jaccard](#jaccard)
			- [Manhattan](#manhattan)
			- [Minkowski](#minkowski)
		- [SVM](#svm)
			- [Linear](#linear)
			- [Polynomial](#polynomial)
			- [RBF](#rbf)
			- [Sigmoidal](#sigmoidal)
	- [Cross Validation](#cross-validation)
		- [Validators](#validators)
			- [Hold Out](#hold-out)
			- [K Fold](#k-fold)
			- [Leave P Out](#leave-p-out)
			- [Monte Carlo](#monte-carlo)
		- [Metrics](#validation-metrics)
			- [Accuracy](#accuracy)
			- [Completeness](#completeness)
			- [F Beta](#f-beta)
			- [Homogeneity](#homogeneity)
			- [MCC](#mcc)
			- [Mean Absolute Error](#mean-absolute-error)
			- [Mean Squared Error](#mean-squared-error)
			- [Median Absolute Error](#median-absolute-error)
			- [Rand Index](#rand-index)
			- [RMS Error](#rms-error)
			- [R Squared](#r-squared)
			- [V Measure](#v-measure)
		- [Reports](#reports)
			- [Aggregate Report](#aggregate-report)
			- [Confusion Matrix](#confusion-matrix)
			- [Contingency Table](#contingency-table)
			- [Multiclass Breakdown](#multiclass-breakdown)
			- [Residual Analysis](#residual-analysis)
	- [Other](#other)
		- [Strategies](#strategies)
			- [Blurry Percentile](#blurry-percentile)
			- [Constant](#constant)
			- [K Most Frequent](#k-most-frequent)
			- [Lottery](#lottery)
			- [Mean](#mean)
			- [Popularity Contest](#popularity-contest)
			- [Wild Guess](#wild-guess)
		- [Helpers](#helpers)
			- [Data Type](#data-type)
			- [Params](#params)
		- [Loggers](#loggers)
			- [Screen](#screen)
		- [Tokenizers](#tokenizers)
			- [N-Gram](#n-gram)
			- [Skip-Gram](#skip-gram)
			- [Whitespace](#whitespace)
			- [Word](#word-tokenizer)
- [FAQ](#faq)
	- [What environment (SAPI) should I run Rubix in?](#what-environment-sapi-should-i-run-rubix-in)
	- [I'm getting out of memory errors](#im-getting-out-of-memory-errors)
    - [What is a Tuple?](#what-is-a-tuple)
    - [What is the difference between categorical and continuous data types?](#what-is-the-difference-between-categorical-and-continuous-data-types)
    - [Does Rubix support multiprocessing?](#does-rubix-support-multiprocessing)
	- [Does Rubix support multithreading?](#does-rubix-support-multi-threading)
	- [Does Rubix support Deep Learning?](#does-rubix-support-deep-learning)
	- [Does Rubix support Reinforcement Learning?](#does-rubix-support-reinforcement-learning)
- [Testing](#testing)
- [Contributing](#contributing)

---
### Basic Introduction
Machine learning (ML) is the process by which a computer program is able to progressively improve performance on a certain task through training and data without explicitly being programmed. There are two types of machine learning that Rubix supports out of the box - *Supervised* and *Unsupervised*.

 - **Supervised** learning is a technique that uses a labeled dataset in which the outcome of each sample has been *labeled* by a human expert prior to training. There are two types of supervised learning to consider in Rubix:
	 - **Classification** is the problem of identifying which *class* a particular sample belongs to. For example, one task may be in determining a particular species of flower or predicting someone's MBTI personality type.
	 - **Regression** aims at predicting *continuous* values such as the sale price of a house or the position (in degrees) of the steering wheel of an automobile. The major difference between classification and regression is that, while there are a *finite* number of classes that a sample can belong to, there are *infinitely* many real (continuous) values that are possible to predict.
- **Unsupervised** learning by contrast does *not* use a labeled dataset. Instead, it focuses on finding patterns within the raw samples.
	- **Clustering** is the grouping of data points in such a way that members of the same group are more similar (homogeneous) than the rest of the samples. You can think of clustering as assigning a class label to an otherwise unlabeled sample. An example where clustering is used is in differentiating PET scan tissues or segmenting a customer base.
	- **Anomaly Detection** is the process of flagging samples that appear to be generated from a mechanism other than one that produces nominal data. Anomalous samples can indicate adversarial activity or exceptional circumstances such as fraud or a cyber attack.
	- **Manifold Learning** is used in visualizing high dimensional datasets, embedding sparse feature representations, and reducing model size by producing a low dimensional representation of the original feature space.

### Obtaining Data
Machine learning projects typically begin with a question. For example, you might want to answer the question "who of my friends are most likely to stay married to their partner?" One way to go about answering this question with machine learning would be to go out and ask a bunch of happily married and divorced couples the same set of questions about their partner and then use that data to build a model of what a successful marriage looks like. Later, you can use that model to make predictions based on the answers you get from your friends. Specifically, the answers you collect are called *features* and they constitute measurements of some phenomena being observed. The number of features in a sample is called the *dimensionality* of the sample. For example, a sample with 20 features is said to be *20 dimensional*.

An alternative to collecting data yourself is to download one of the many open datasets that are free to use from a public repository. The advantages of using a public dataset is that, usually, the data has already been cleaned and prepared for you. We recommend the University of California Irvine [Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets.php) as a great place to get started with using open source datasets.

There are many PHP libraries that help make extracting data from various sources such as CSV, database, and the cloud easy and intuitive, and we recommend checking those out as well.

Here are some libraries that we recommend for data extraction:

- [PHP League CSV](https://csv.thephpleague.com/) - Generator-based CSV extractor
- [Doctrine DBAL](https://www.doctrine-project.org/projects/dbal.html) - SQL database abstraction layer
- [Google BigQuery](https://cloud.google.com/bigquery/docs/reference/libraries) - Cloud-based data warehouse via SQL

#### The Dataset Object
In Rubix, data is passed around in specialized data containers called Datasets. [Dataset objects](#dataset-objects) internally handle selecting, splitting, folding, transforming, and randomizing the samples and labels contained within. In general, there are two types of datasets, *Labeled* and *Unlabeled*. Labeled datasets are used for *supervised* learning and for providing the ground-truth during cross validation. Unlabeled datasets are used for *unsupervised* learning and for making predictions (which we call *inference*) on unknown samples.

For the following example, suppose that you went out and asked 100 couples (50 married and 50 divorced) about their partner's communication skills (between 1 and 5), attractiveness (between 1 and 5), and time spent together per week (hours per week). You could construct a [Labeled Dataset](#labeled) from this data like so:

```php
use Rubix\ML\Datasets\Labeled;

$samples = [
    [3, 4, 50.5], [1, 5, 24.7], [4, 4, 62.0], [3, 2, 31.1], ...
];

$labels = ['married', 'divorced', 'married', 'divorced', ...];

$dataset = new Labeled($samples, $labels);
```

### Choosing an Estimator
Estimators make up the core of the Rubix library as they are responsible for making predictions. There are many different algorithms to choose from and each one performs differently. Choosing the right [Estimator](#estimators) for the job is crucial to creating a system that balances accuracy and performance.

In practice, you will test out a number of different estimators to get the best sense of what works for your particular dataset. However, for our example problem we will just focus on a simple classifier called [K Nearest Neighbors](#k-nearest-neighbors). Since the label of each training sample we collect will be a discrete class (*married couples* or *divorced couples*), we need an Estimator that is designed to output class predictions. The K Nearest Neighbors classifier works by locating the closest training samples to an unknown sample and choosing the class label that appears most often.

#### Creating the Estimator Instance

Like most Estimators, the K Nearest Neighbors (KNN) classifier requires a set of parameters (called *hyper-parameters*) to be chosen up front by the user. These parameters can be selected based on some prior knowledge of the problem space, or at random. The defaults provided in Rubix are a good place to start for most machine learning problems. In addition, the library provides a meta-Estimator called [Grid Search](#grid-search) that searches for the best combination of hyper-parmeters for a particular estimator given a set of possible values. For the purposes of our simple example we will just go with our intuition and choose the parameters outright.

> **Note**: You can find a full description of all of the K Nearest Neighbors hyper-parameters in the [API reference](#k-nearest-neighbors) guide.

In KNN, the hyper-parameter *k* is the number of nearest points from the training set to compare an unknown sample to in order to infer its class label. For example, if the 5 closest neighbors to a given unknown sample have 4 married labels and 1 divorced label, then the algorithm will output a prediction of married with a probability of 0.8.

The second hyper-parameter is the distance *kernel* that determines how distance is measured within the model. We'll go with standard [Euclidean](#euclidean) distance for now.

To instantiate a K Nearest Neighbors classifier:
```php
use Rubix\ML\Classifiers\KNearestNeighbors;
use Rubix\ML\Kernels\Distance\Euclidean;

// Using the default hyper-parameters
$estimator = new KNearestNeighbors();

// Specifying the hyper-parameters
$estimator = new KNearestNeighbors(5, new Euclidean());
```

Now that we've chosen and instantiated our estimator and our Dataset object is ready to go, we are ready to train the model and use it to make some predictions.

### Training and Prediction
Training is the process of feeding the learning algorithm data so that it can build a model of the problem. A trained model consists of all of the parameters (except hyper-parameters) that are required for the estimator to make predictions. If you try to make predictions using an untrained learner, it will throw an exception.

Passing the Labeled dataset to the instantiated learner, we can train our K Nearest Neighbors classifier like so:
```php
$estimator->train($dataset);
```

We can verify that the learner has been trained by calling the `trained()` method:
```php
var_dump($estimator->trained());
```

**Output:**

```sh
bool(true)
```

For our 100 sample example training set, training should only take a matter of microseconds, but larger datasets with higher dimensionality and fancier learning algorithms can take much longer. Once the estimator has been fully trained, we can now feed in some unknown samples to see what the model predicts.

Turning back to our example problem, suppose that we went out and collected 5 new data points from our friends using the same questions we asked the couples we interviewed for our training set. We could make predictions on whether they will stay married or get divorced by taking their answers as features and running them in an Unlabeled dataset through the trained Estimator's `predict()` method.
```php
use Rubix\ML\Dataset\Unlabeled;

$unknown = [
    [4, 3, 44.2], [2, 2, 16.7], [2, 4, 19.5], [1, 5, 8.6], [3, 3, 55.0],
];

$dataset = new Unlabeled($unknown);

$predictions = $estimator->predict($dataset);

var_dump($predictions);
```

**Output:**

```sh
array(5) {
	[0] => 'married'
	[1] => 'divorced'
	[2] => 'divorced'
	[3] => 'divorced'
	[4] => 'married'
}
```

### Evaluating Model Performance
Making predictions is not very useful unless the estimator can correctly generalize what it has learned during training to the real world. [Cross Validation](#cross-validation) is a process by which we can test the model for its generalization ability. For the purposes of this introduction, we will use a simple form of cross validation called *Hold Out*. The [Hold Out](#hold-out) validator will take care of splitting the dataset into training and testing sets automatically, such that a portion of the data is *held out* to be used for testing (or *validating*) the model. The reason we do not use *all* of the data for training is because we want to test the Estimator on samples that it has never seen before.

The Hold Out validator requires you to set the ratio of testing to training samples as a constructor parameter. In this case, let's choose to use a factor of 0.2 (20%) of the dataset for testing leaving the rest (80%) for training. Typically, 0.2 is a good default choice however your mileage may vary. The important thing to note here is the trade off between more data for training and more data to produce precise testing results. Once you get the hang of Hold Out, the next step is to consider more advanced cross validation techniques such as [K Fold](#k-fold), [Leave P Out](#leave-p-out), and [Monte Carlo](#monte-carlo) simulations.

To return a score from the Hold Out validator using the Accuracy metric just pass it the untrained estimator instance and a dataset:

```php
use Rubix\ML\CrossValidation\HoldOut;
use Rubix\ML\CrossValidation\Metrics\Accuracy;

$validator = new HoldOut(0.2);

$score = $validator->test($estimator, $dataset, new Accuracy());

var_dump($score);
```

**Output:**

```sh
float(0.945)
```

### Visualization
Visualization is how you communicate the findings of your experiment to the end-user and is key to deriving value from your hard work. Although visualization is important (important enough for us to mention it), we consider it to be beyond the scope of Rubix . Therefore, we leave you with the freedom to chose one of the many great plotting and visualization applications out there to communicate the insights you obtain using Rubix.

If you are just looking for a quick way to visualize data then we recommend exporting it to a file (JSON or CSV for example) and importing it into your favorite plotting or spreadsheet software such as [Plotly](https://plot.ly/), [Tableu](https://www.tableau.com), [Google Sheets](https://www.google.com/sheets/about/), or [Excel](https://products.office.com/en-us/excel). PHP has built in functions for manipulating both JSON and CSV formats, and there are a number of libraries available that help reading and writing these formats to file from PHP.

If you are looking to publish your visualizations to the world, we highly recommend [D3.js](https://d3js.org/) since it is an amazing data-driven framework written in Javascript that plays well with PHP.

### Next Steps
After you've gone through this basic introduction to machine learning, we highly recommend checking out all of the [example projects](https://github.com/RubixML) and reading over the [API Reference](#api-reference) to get a powerful understanding for what you can do with Rubix. If you have a question or need help, feel free to post on our Github page. We'd love to hear from you.

---
### System Architecture
The Rubix architecture is defined by a few key abstractions and their corresponding types and interfaces.

![Rubix ML System Architecture](https://raw.githubusercontent.com/RubixML/RubixML/master/docs/images/rubix-ml-system-architecture.svg?sanitize=true)

---
### API Reference
This section breaks down the public application programming interface (API) of each Rubix ML component in detail.

---
### Interfaces
Interfaces define the basic functionality of every object in Rubix.

### Estimator
Estimators consist of [Classifiers](#classifiers), [Regressors](#regressors), [Clusterers](#clusterers), [Embedders](#embedders), and [Anomaly Detectors](#anomaly-detectors) that make *predictions* based on data. Estimators that can be trained with data are called *Learners* and they can either be supervised or unsupervised depending on the task. Estimators can employ methods on top of the basic API by implementing a number of addon interfaces such as [Online](#online), [Probabilistic](#probabilistic), [Persistable](#persistable), and [Verbose](#verbose). The most basic Estimator is one that outputs an array of predictions given a dataset of unknown or testing samples.

> **Note**: The return value of `predict()` is an array containing the predictions indexed in the same order that they were fed into the estimator.

To make predictions from a dataset object:
```php
public predict(Dataset $dataset) : array
```

**Example:**

```php
$predictions = $estimator->predict($dataset);

var_dump($predictions);
```

**Output:**

```sh
array(3) {
  [0]=>
  string(7) "married"
  [1]=>
  string(8) "divorced"
  [2]=>
  string(7) "married"
}

```

### Learner
Most estimators have the ability to be trained with data. These estimators are called *Learners* and require training before they are able make predictions. Training is the process of feeding data to the learner so that it can formulate a generalized function that maps future samples to good predictions.

> **Note**: Calling `train()` on an already trained estimator will cause any previous training to be lost. If you would like to be able to train a model incrementally, see the [Online](#online) Estimator interface.

To train an learner pass it a training dataset:
```php
public train(Dataset $training) : void
```

Return whether or not the learner has been trained:
```php
public trained() : bool
```

**Example:**
```php
$estimator->train($dataset);

var_dump($estimator->trained());
```

**Output:**
```sh
bool(true)
```

### Online
Certain estimators that implement the *Online* interface can be trained in batches. Estimators of this type are great for when you either have a continuous stream of data or a dataset that is too large to fit into memory. Partial training allows the model to evolve as new information about the world is acquired.

> **Note**: Learner will continue to train as long as you are using the `partial()` method, however, calling `train()` on a trained or partially trained learner will reset it back to baseline first.

To partially train an online learner:
```php
public partial(Dataset $dataset) : void
```

**Example:**
```php
$folds = $dataset->fold(3);

$estimator->train($folds[0]);

$estimator->partial($folds[1]);

$estimator->partial($folds[2]);
```

### Parallel
Multiprocessing is the use of two or more processes that *usually* execute in parallel on multiple cores. Objects that implement the Parallel interface can take advantage of multi core systems by executing parts or all of their algorithm in parallel. Parallelizable objects can utilize various parllel processing [Backends](#backends) under the hood which are set using the `setBackend()` method.

> **Note**: The optimal number of workers will depend on the system specifications of the computer. Fewer workers than CPU cores may not achieve full processing potential but more workers than cores can cause excess overhead.

> **Note**: Unless otherwise stated, all objects implementing Parallel have a default backend of [Serial](#serial).

To set the backend processing engine:
```php
public setBackend(Backend $backend) : void
```

**Example:**

```php
use Rubix\ML\Backends\Amp;

$estimator->setBackend(new Amp(4));
```

### Persistable
If an estimator implements Persistable then it can be saved and loaded by a [Persister](#persisters) or by wrapping it with a [Persistent Model](#persistent-model) meta estimator.

### Probabilistic
Estimators that implement the *Probabilistic* interface have an additional method that returns an array of probability scores of each possible class, cluster, etc. Probabilities are useful for ascertaining the degree to which the estimator is certain about a particular prediction.

Return the probability estimates of a prediction:
```php
public proba(Dataset $dataset) : array
```

**Example:**
```php
$probabilities = $estimator->proba($dataset);  

var_dump($probabilities);
```

**Output:**
```sh
array(2) {
	[0] => array(2) {
		['married'] => 0.975,
		['divorced'] => 0.025,
	}
	[1] => array(2) {
		['married'] => 0.200,
		['divorced'] => 0.800,
	}
}
```

### Ranking
In the way that a Probabilistic estimator ranks the outcome of a particlar sample as a *normalized* (between 0 and 1) value, Ranking estimators rank the outcome by an *arbitrary* score. The purpose of a Ranking estimator is so that you are able to sort the samples by the output. This is useful in cases such as [Anomaly Detection](#anomaly-detectors) where an analyst can flag the top n outliers by rank for further investigation.

To rank the dataset by an artitrary scoring function:
```php
public rank(Dataset $dataset) : array
```

**Example:**
```php
$scores = $estimator->rank($dataset);

var_dump($scores);
```

**Output:**
```sh
array(3) {
  [0]=> float(1.80)
  [1]=> int(1.25)
  [2]=> int(9.45)
}
```

### Verbose
Verbose objects are capable of logging important events to any PSR-3 compatible logger such as [Monolog](https://github.com/Seldaek/monolog), [Analog](https://github.com/jbroadway/analog), or the included [Screen Logger](#screen). Logging is especially useful for monitoring the progress of the underlying learning algorithm in real time.

To set the logger pass in any PSR-3 compatible logger instance:
```php
public setLogger(LoggerInterface $logger) : void
```

**Example:**
```php
use Rubix\ML\Other\Loggers\Screen;

$estimator->setLogger(new Screen('sentiment'));
```

### Wrapper
Wrappers are meta-estimators that wrap a base estimator for the purposes of adding extra functionality. Most wrappers allow access to the underlying base estimator's methods from the Wrapper instance, but you can also return the base estimator directly using the `base()` method.

To return the base estimator:
```php
public base() : Estimator
```

**Example:**

```php
$base = $estimator->base();
```

### Dataset Objects
In Rubix, data are passed around using specialized container structures called Dataset objects. Dataset objects can hold a heterogeneous mix of categorical and continuous data and make it easy to transport data in a canonical way. 

> **Note**: There are two *types* of features that estimators can process i.e *categorical* and *continuous*. Any numerical (integer or float) datum is considered continuous and any string datum is considered categorical by convention throughout Rubix.

The Dataset interface has an assorted API designed to make working on datasets fast and easy. Below you'll find a description of the various methods available on the basic interface.

#### Stacking
Stack a number of dataset objects on top of each other and return a single dataset:
```php
public static stack(array $datasets) : self;
```

#### Selecting
Return all the samples in the dataset:
```php
public samples() : array
```

Select the *sample* at row offset:
```php
public row(int $index) : array
```

Select the *values* of a feature column at offset:
```php
public column(int $index) : array
```

Return the *first* **n** rows of data in a new dataset object:
```php
public head(int $n = 10) : self
```

Return the *last* **n** rows of data in a new dataset object:
```php
public tail(int $n = 10) : self
```

**Example:**

```php
// Return the sample matrix
$samples = $dataset->samples();

// Return just the first 5 rows in a new dataset
$subset = $dataset->head(5);
```

#### Properties

Return the number of rows in the dataset:
```php
public numRows() : int
```

Return the number of columns in the dataset:
```php
public numColumns() : int
```

Return the integer encoded column types for each feature column:
```php
public types() : array
```

Return the integer encoded column type given a column index:
```php
public columnType(int $index) : int
```

Return the range for each feature column:
```php
public ranges() : array
```

Return the range of a feature column. The range for a continuous column is defined as the minimum and maximum values, and for categorical columns the range is defined as every unique category.
```php
public columnRange(int $index) : array
```

#### Splitting, Folding, and Batching

Remove **n** rows from the dataset and return them in a new dataset:
```php
public take(int $n = 1) : self
```

Leave **n** samples on the dataset and return the rest in a new dataset:
```php
public leave(int $n = 1) : self
```

Split the dataset into *left* and *right* subsets given by a **ratio**:
```php
public split(float $ratio = 0.5) : array
```

Partition the dataset into *left* and *right* subsets based on the value of a feature in a specified column:
```php
public partition(int $index, mixed $value) : array
```

Fold the dataset **k** - 1 times to form **k** equal size datasets:
```php
public fold(int $k = 10) : array
```

Batch the dataset into subsets of **n** rows per batch:
```php
public batch(int $n = 50) : array
```

**Example:**

```php
// Remove the first 5 rows and return them in a new dataset
$subset = $dataset->take(5);

// Split the dataset into left and right subsets
[$left, $right] = $dataset->split(0.5);

// Partition the dataset by the feature column at index 4 by value '50'
[$left, $right] = $dataset->partition(4, 50);

// Fold the dataset into 8 equal size datasets
$folds = $dataset->fold(8);
```

#### Randomizing

Randomize the order of the Dataset and return it:
```php
public randomize() : self
```

Generate a random subset with replacement of size **n**:
```php
public randomSubsetWithReplacement($n) : self
```

Generate a random *weighted* subset with replacement of size **n**:
```php
public randomWeightedSubsetWithReplacement($n, array $weights) : self
```

**Example:**
```php
// Randomize and split the dataset into two subsets
[$left, $right] = $dataset->randomize()->split(0.8);

// Generate a bootstrap dataset of 500 random samples
$subset = $dataset->randomSubsetWithReplacement(500);
```

#### Filtering

To filter a Dataset by a feature column:
```php
public filterByColumn(int $index, callable $fn) : self
```

**Example:**
```php
$tallPeople = $dataset->filterByColumn(2, function ($value) {
	return $value > 178.5;
});
```

#### Sorting

To sort a dataset in place by a specific feature column:
```php
public sortByColumn(int $index, bool $descending = false) : self
```

**Example:**
```php
var_dump($dataset->samples());

$dataset->sortByColumn(2, false);

var_dump($dataset->samples());
```

**Output:**
```sh
array(3) {
    [0]=> array(3) {
	    [0]=> string(4) "mean"
	    [1]=> string(4) "furry"
	    [2]=> int(8)
    }
    [1]=> array(3) {
	    [0]=> string(4) "nice"
	    [1]=> string(4) "rough"
	    [2]=> int(1)
    }
    [2]=> array(3) {
	    [0]=> string(4) "nice"
	    [1]=> string(4) "rough"
	    [2]=> int(6)
    }
}

array(3) {
    [0]=> array(3) {
	    [0]=> string(4) "nice"
	    [1]=> string(4) "rough"
	    [2]=> int(1)
    }
    [1]=> array(3) {
	    [0]=> string(4) "nice"
	    [1]=> string(4) "rough"
	    [2]=> int(6)
    }
    [2]=> array(3) {
	    [0]=> string(4) "mean"
	    [1]=> string(4) "furry"
	    [2]=> int(8)
    }
}
```

#### Prepending and Appending
To prepend a given dataset onto the beginning of another dataset:
```php
public prepend(Dataset $dataset) : self
```

To append a given dataset onto the end of another dataset:
```php
public append(Dataset $dataset) : self
```

#### Applying a Transformation
You can apply a fitted [Transformer](#transformers) to a Dataset directly passing it to the apply method on the Dataset.

```php
public apply(Transformer $transformer) : void
```

**Example:**
```php
use Rubix\ML\Transformers\OneHotEncoder;

$transformer = new OneHotEncoder();

$transformer->fit($dataset);

$dataset->apply($transformer);
```

### Labeled
For *supervised* Estimators you will need to train it with a Labeled dataset consisting of samples with the addition of labels that correspond to the observed outcome of each sample. Splitting, folding, randomizing, sorting, and subsampling are all done while keeping the indices of samples and labels aligned. In addition to the basic Dataset interface, the Labeled class can sort and *stratify* the data by label as well.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Datasets/Labeled.php)

**Parameters:**
| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | samples | | array | A 2-dimensional array consisting of rows of samples and columns with feature values. |
| 2 | labels | | array | A 1-dimensional array of labels that correspond to the samples in the dataset. |
| 3 | validate | true | bool | Should we validate the data? |

**Additional Methods:**
Build a new labeled dataset with validation:
```php
public static build(array $samples = [], array $labels = []) : self
```

Build a new labeled dataset foregoing validation:
```php
public static quick(array $samples = [], array $labels = []) : self
```

Build a dataset using a pair of iterators:
```php
public static fromIterator(iterable $samples, iterable $labels) : self
```

Return an array of labels:
```php
public labels() : array
```

Return the samples and labels in a single array:
```php
public zip() : array
```

Return the label at the given row offset:
```php
public label(int $index) : mixed
```

Return the type of the label encoded as an integer:
```php
public labelType() : int
```

Return all of the possible outcomes i.e. the unique labels:
```php
public possibleOutcomes() : array
```

Transform the labels in the dataset using a function:
```php
public transformLabels(callable $fn) : void
```

The function is given a label as its only argument and should return the new label as a continuous or categorical value.

```php
$dataset->transformLabels(function ($label) {
	return $label === 1 ? 'female' : 'male';
});
```

Filter the dataset by label:
```php
public filterByLabel(callable $fn) : self
```

Sort the dataset by label:
```php
public sortByLabel(bool $descending = false) : self
```

Group the samples by label and return them in their own dataset:
```php
public stratify() : array
```

Split the dataset into left and right stratified subsets with a given *ratio* of samples in each:
```php
public stratifiedSplit($ratio = 0.5) : array
```

Return *k* equal size subsets of the dataset:
```php
public stratifiedFold($k = 10) : array
```

**Example:**
```php
use Rubix\ML\Datasets\Labeled;

// Import samples and labels

$dataset = Labeled::build($samples, $labels);  // Build a new dataset with validation

// or ...

$dataset = Labeled::quick($samples, $labels);  // Build a new dataset without validation

// or ...

$dataset = new Labeled($samples, $labels, true);  // Use the full constructor

// Transform integer encoded labels to strings
$dataset->transformLabels(function ($label) {
	switch ($label) {
		case 1:
			return 'male';
		
		case 2:
			return 'female';

		default:
			return 'unknown';
	}
});

// Return all the labels in the dataset
$labels = $dataset->labels();

// Return the label at offset 3
$label = $dataset->label(3);

// Return all possible unique labels
$outcomes = $dataset->possibleOutcomes();

var_dump($labels);
var_dump($label);
var_dump($outcomes);
```

**Output:**
```sh
array(4) {
    [0]=> string(5) "female"
    [1]=> string(4) "male"
    [2]=> string(5) "female"
    [3]=> string(4) "male"
}

string(4) "male"

array(2) {
	[0]=> string(5) "female"
	[1]=> string(4) "male"
}
```

**Example:**
```php
// Fold the dataset into 5 equal size stratified subsets
$folds = $dataset->stratifiedFold(5);

// Split the dataset into two stratified subsets
[$left, $right] = $dataset->stratifiedSplit(0.8);

// Put each sample with label x into its own dataset
$strata = $dataset->stratify();
```

### Unlabeled
Unlabeled datasets can be used to train *unsupervised* Estimators and for feeding data into an Estimator to make predictions.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Datasets/Unlabeled.php)

**Parameters:**
| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | samples | | array | A 2-dimensional array consisting of rows of samples and columns with feature values. |
| 2 | validate | true | bool | Should we validate the input? |


**Additional Methods:**
Build a new unlabeled dataset with validation:
```php
public static build(array $samples = []) : self
```

Build a new unlabeled dataset foregoing validation:
```php
public static quick(array $samples = []) : self
```

Build a dataset with an iterator:
```php
public static fromIterator(iterable $samples) : self
```

**Example:**
```php
use Rubix\ML\Datasets\Unlabeled;

$dataset = Unlabeled::build($samples);  // Build a new dataset with validation

// or ...

$dataset = Unlabeled::quick($samples);  // Build a new dataset without validation

// or ...

$dataset = new Unlabeled($samples, true);  // Use the full constructor
```

### Generators
Dataset generators produce synthetic datasets of a user-specified shape, dimensionality, and cardinality. Synthetic data is useful for a number of tasks including experimenting with data of various shapes, augmenting an already existing dataset with more data, or for testing and demonstration purposes.

To generate a Dataset object with **n** samples (*rows*):
```php
public generate(int $n) : Dataset
```

Return the dimensionality of the samples produced by the generator:
```php
public dimensions() : int
```

**Example:**

```php
use Rubix\ML\Datasets\Generators\Blob;

$generator = new Blob([0, 0], 1.0);

$dataset = $generator->generate(3);

var_dump($generator->dimensions());

var_dump($dataset->samples());
```

**Output:**

```sh
int(2)

object(Rubix\ML\Datasets\Unlabeled)#24136 (1) {
  ["samples":protected]=>
  array(3) {
    [0]=>
    array(2) {
      [0]=> float(-0.2729673885539)
      [1]=> float(0.43761840244204)
    }
    [1]=>
    array(2) {
      [0]=> float(-1.2718092282012)
      [1]=> float(-1.9558245484829)
    }
    [2]=>
    array(2) {
      [0]=> float(1.1774185431405)
      [1]=> float(0.05168623824664)
    }
  }
}
```

### Agglomerate
An Agglomerate is a collection of generators with each of them given a user-defined label. Agglomerates are useful for classification, clustering, and anomaly detection problems where the target label is a discrete value.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Datasets/Generators/Agglomerate.php)

**Data Types:** Depends on agglomerated generators' types

**Label Type:** Categorical

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | generators | | array | A collection of generators keyed by their user-specified label (0 indexed by default). |
| 2 | weights | Auto | array | A set of arbitrary weight values corresponding to a generator's proportion to the overall agglomeration. |

**Additional Methods:**

Return the normalized weight values of each generator in the agglomerate:
```php
public weights() : array
```

**Example:**

```php
use Rubix\ML\Datasets\Generators\Agglomerate;

$generator = new Agglomerate([
	new Blob([5, 2], 1.0),
	new HalfMoon(-3, 5, 1.5, 90.0, 0.1),
	new Circle(2, -4, 2.0, 0.05),
], [
	5, 6, 3, // Arbitrary weight values
]);
```

### Blob
A normally distributed (Gaussian) n-dimensional blob of samples centered at a given vector. The standard deviation can be set for the whole blob or for each feature column independently. When a global value is used, the resulting blob will be isotropic and will converge asypmtotically to a sphere.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Datasets/Generators/Blob.php)

**Data Types:** Continuous

**Label Type:** None

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | center | [0.0, 0.0] | array | The coordinates of the center of the blob. |
| 2 | stddev | 1.0 | float or array | Either the global standard deviation or an array with the standard deviation on a per feature column basis. |

**Additional Methods:**

This generator does not have any additional methods.

**Example:**

```php
use Rubix\ML\Datasets\Generators\Blob;

$generator = new Blob([-1.2, -5., 2.6, 0.8, 10.], 0.25);
```

### Circle
Creates a dataset of points forming a circle in 2 dimensions. The label of each sample is the random value used to generate the projection measured in degrees.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Datasets/Generators/Circle.php)

**Data Types:** Continuous

**Label Type:** Continuous

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | x | 0.0 | float | The *x* coordinate of the center of the circle. |
| 2 | y | 0.0 | float | The *y* coordinate of the center of the circle. |
| 3 | scale | 1.0 | float | The scaling factor of the circle. |
| 4 | noise | 0.1 | float | The amount of Gaussian noise to add to each data point as a ratio of the scaling factor. |

**Additional Methods:**

This generator does not have any additional methods.

**Example:**

```php
use Rubix\ML\Datasets\Generators\Circle;

$generator = new Circle(0.0, 0.0, 100, 0.1);
```

### Half Moon
Generate a dataset consisting of 2 dimensional samples that form a half moon shape when plotted on a chart. The label for each sample is the value obtained by reversing the generative process for that particular sample.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Datasets/Generators/HalfMoon.php)

**Data Types:** Continuous

**Label Type:** Continuous

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | x | 0.0 | float | The *x* coordinate of the center of the half moon. |
| 2 | y | 0.0 | float | The *y* coordinate of the center of the half moon. |
| 3 | scale | 1.0 | float | The scaling factor of the half moon. |
| 4 | rotate | 90.0 | float | The amount in degrees to rotate the half moon counterclockwise. |
| 5 | noise | 0.1 | float | The amount of Gaussian noise to add to each data point as a percentage of the scaling factor. |

**Additional Methods:**

This generator does not have any additional methods.

**Example:**

```php
use Rubix\ML\Datasets\Generators\HalfMoon;

$generator = new HalfMoon(4.0, 0.0, 6, 180.0, 0.2);
```

### Swiss Roll
Generate a non-linear 3-dimensional dataset resembling a *swiss roll* or spiral. The labels are the seeds to the swiss roll transformation.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Datasets/Generators/SwissRoll.php)

**Data Types:** Continuous

**Label Type:** Continuous

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | x | 0.0 | float | The *x* coordinate of the center of the swiss roll. |
| 2 | y | 0.0 | float | The *y* coordinate of the center of the swiss roll. |
| 3 | z | 0.0 | float | The *z* coordinate of the center of the swiss roll. |
| 4 | scale | 1.0 | float | The scaling factor of the swiss roll. |
| 5 | depth | 21.0 | float | The depth of the swiss roll i.e the scale of the y axis. |
| 6 | noise | 0.3 | float | The standard deviation of the gaussian noise. |

**Additional Methods:**

This generator does not have any additional methods.

**Example:**

```php
use Rubix\ML\Datasets\Generators\SwissRoll;

$generator = new SwissRoll(5.5, 1.5, -2.0, 10, 21.0, 0.2);
```

---
### Anomaly Detectors
Anomaly detection is the process of identifying samples that do not conform to an expected pattern. The output prediction of a detector is a binary encoding (either *0* for a normal sample or *1* for a detected anomaly).

### Isolation Forest
An ensemble detector comprised of Isolation Trees each trained on a different subset of the training set. The Isolation Forest works by averaging the isolation score of a sample across a user-specified number of trees.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/AnomalyDetectors/IsolationForest.php)

**Interfaces:** [Estimator](#estimators), [Learner](#learner), [Ranking](#ranking), [Persistable](#persistable)

**Compatibility:** Categorical, Continuous

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | estimators | 300 | int | The number of estimators to train in the ensemble. |
| 2 | contamination | 0.1 | float | The percentage of outliers that are assumed to be present in the training set. |
| 3 | ratio | 0.2 | float | The ratio of random samples to train each estimator with. |

**Additional Methods:**

This estimator does not have any additional methods.

**Example:**

```php
use Rubix\ML\AnomalyDetection\IsolationForest;

$estimator = new IsolationForest(300, 0.01, 0.2);
```

**References:**
>- F. T. Liu et al. (2008). Isolation Forest.
>- F. T. Liu et al. (2011). Isolation-based Anomaly Detection.

### K-d LOF
A k-d tree accelerated version of [Local Outlier Factor](#local-outlier-factor) which benefits from neighborhood pruning during nearest neighbors search. The tradeoff between K-d LOF and the brute force method is that while K-d LOF is faster, it cannot be partially trained.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/AnomalyDetectors/KDLOF.php)

**Interfaces:** [Estimator](#estimators), [Learner](#learner), [Ranking](#ranking), [Persistable](#persistable)

**Compatibility:** Continuous

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | k | 20 | int | The k nearest neighbors that form a local region. |
| 2 | contamination | 0.1 | float | The percentage of outliers that are assumed to be present in the training set. |
| 3 | kernel | Euclidean | object | The distance kernel used to compute the distance between sample points. |
| 4 | max leaf size | 30 | int | The max number of samples in a leaf node (*neighborhood*). |

**Additional Methods:**

Return the base k-d tree instance:
```php
public tree() : KDTree
```

**Example:**

```php
use Rubix\ML\AnomalyDetection\KDLOF;
use Rubix\ML\Kernels\Distance\Euclidean;

$estimator = new KDLOF(20, 0.1, new Euclidean(), 30);
```

**References:**

>- M. M. Breunig et al. (2000). LOF: Identifying Density-Based Local Outliers.

### LODA
Lightweight Online Detector of Anomalies uses sparse random projection vectors to produce an ensemble of unique one dimensional equi-width histograms able to estimate the probability density of an unknown sample. The anomaly score is given by the negative log likelihood whose upper threshold can be set by the user.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/AnomalyDetectors/LODA.php)

**Interfaces:** [Estimator](#estimators), [Learner](#learner), [Online](#online), [Ranking](#ranking), [Persistable](#persistable)

**Compatibility:** Continuous

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | bins | 5 | int | The number of equi-width bins for each histogram. |
| 2 | estimators | 100 | int | The number of random projections and histograms. |
| 3 | threshold | 5.5 | float | The threshold anomaly score to be flagged as an outlier. |

**Additional Methods:**

To estimate the number of histogram bins from a dataset:
```php
public static estimateBins($dataset) : int
```

**Example:**

```php
use Rubix\ML\AnomalyDetection\LODA;

$bins = LODA::estimateBins($dataset);

$estimator = new LODA($bins, 250, 3.5);
```

**References:**

>- T. Pevný. (2015). Loda: Lightweight on-line detector of anamalies.
>- L. Birg´e et al. (2005). How Many Bins Should Be Put In A Regular Histogram.

### Local Outlier Factor
Local Outlier Factor (LOF) measures the local deviation of density of a given sample with respect to its *k* nearest neighbors. As such, LOF only considers the local region (or *neighborhood*) of an unknown sample which enables it to detect anomalies within individual clusters of data.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/AnomalyDetectors/LocalOutlierFactor.php)

**Interfaces:** [Estimator](#estimators), [Learner](#learner), [Online](#online), [Ranking](#ranking), [Persistable](#persistable)

**Compatibility:** Continuous

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | k | 20 | int | The k nearest neighbors that form a local region. |
| 2 | contamination | 0.1 | float | The percentage of outliers that are assumed to be present in the training set. |
| 3 | kernel | Euclidean | object | The distance kernel used to compute the distance between sample points. |

**Additional Methods:**

This estimator does not have any additional methods.

**Example:**

```php
use Rubix\ML\AnomalyDetection\LocalOutlierFactor;
use Rubix\ML\Kernels\Distance\Minkowski;

$estimator = new LocalOutlierFactor(20, 0.1, new Minkowski(3.5));
```

**References:**

>- M. M. Breunig et al. (2000). LOF: Identifying Density-Based Local Outliers.

### One Class SVM
An unsupervised Support Vector Machine used for anomaly detection. The One Class SVM aims to find a maximum margin between a set of data points and the *origin*, rather than between classes such as with  multiclass SVM ([SVC](#svc)).

> **Note**: This estimator requires the [SVM PHP extension](https://php.net/manual/en/book.svm.php) which uses the LIBSVM engine written in C++ under the hood.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/AnomalyDetectors/OneClassSVM.php)

**Interfaces:** [Learner](#learner), [Persistable](#persistable)

**Compatibility:** Continuous

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | nu | 0.1 | float | An upper bound on the percentage of margin errors and a lower bound on the percentage of support vectors. |
| 2 | kernel | RBF | object | The kernel function used to express non-linear data in higher dimensions. |
| 3 | shrinking | true | bool | Should we use the shrinking heuristic? |
| 4 | tolerance | 1e-3 | float | The minimum change in the cost function necessary to continue training. |
| 5 | cache size | 100. | float | The size of the kernel cache in MB. |

**Additional Methods:**

This estimator does not have any additional methods.

**Example:**

```php
use Rubix\ML\AnomalyDetection\OneClassSVM;
use Rubix\ML\Kernels\SVM\Polynomial;

$estimator = new OneClassSVM(0.1, new Polynomial(4), true, 1e-3, 100.);
```

**References:**

>- C. Chang et al. (2011). LIBSVM: A library for support vector machines.

### Robust Z Score
A quick *global* anomaly detector that uses a modified Z score which is robust to outliers to detect anomalies within a dataset. The modified Z score consists of taking the median and median absolute deviation (MAD) instead of the mean and standard deviation (*standard* Z score) thus making the statistic more robust to training sets that may already contain outliers. Outliers can be flagged in one of two ways. First, their average Z score can be above the user-defined tolerance level or an individual feature's score could be above the threshold (*hard* limit).

> [Source](https://github.com/RubixML/RubixML/blob/master/src/AnomalyDetectors/RobustZScore.php)

**Interfaces:** [Estimator](#estimators), [Learner](#learner), [Persistable](#persistable)

**Compatibility:** Continuous

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | tolerance | 3.0 | float | The average z score to tolerate before a sample is considered an outlier. |
| 2 | threshold | 3.5 | float | The threshold z score of a individual feature to consider the entire sample an outlier. |

**Additional Methods:**

Return the median of each feature column in the training set:
```php
public medians() : ?array
```

Return the median absolute deviation (MAD) of each feature column in the training set:
```php
public mads() : ?array
```

**Example:**

```php
use Rubix\ML\AnomalyDetection\RobustZScore;

$estimator = new RobustZScore(1.5, 3.0);
```

**References:**

>- P. J. Rousseeuw et al. (2017). Anomaly Detection by Robust Statistics.

---
### Classifiers
Classifiers are a type of estimator that predict discrete outcomes such as categorical class labels.

### AdaBoost
Short for *Adaptive Boosting*, this ensemble classifier can improve the performance of an otherwise *weak* classifier by focusing more attention on samples that are harder to classify.

> **Note**: The default base classifier is a *Decision Stump* i.e a Classification Tree with a max depth of 1.

> [Source] (https://github.com/RubixML/RubixML/blob/master/src/Classifiers/AdaBoost.php)

**Interfaces:** [Estimator](#estimators), [Learner](#learner), [Probabilistic](#probabilistic), [Verbose](#verbose), [Persistable](#persistable)

**Compatibility:** Depends on base learner

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | base | Classification Tree | object | The base *weak* classifier to be boosted. |
| 2 | estimators | 100 | int | The number of estimators to train in the ensemble. |
| 3 | rate | 1.0 | float | The learning rate i.e step size. |
| 4 | ratio | 0.8 | float | The ratio of samples to subsample from the training set per epoch. |
| 5 | tolerance | 1e-3 | float | The amount of validation error to tolerate before an early stop is considered. |

**Additional Methods:**

Return the calculated weight values of the last trained dataset:
```php
public weights() : array
```

Return the influence scores for each boosted classifier:
```php
public influences() : array
```

Return the training error at each epoch:
```php
public steps() : array
```

**Example:**

```php
use Rubix\ML\Classifiers\AdaBoost;
use Rubix\ML\Classifiers\ExtraTreeClassifier;

$estimator = new AdaBoost(new ExtraTreeClassifier(3), 100, 0.1, 0.5, 1e-2);
```

**References:**

 >- Y. Freund et al. (1996). A Decision-theoretic Generalization of On-line Learning and an Application to Boosting.
 >- J. Zhu et al. (2006). Multi-class AdaBoost.

### Classification Tree
A binary tree-based classifier that minimizes gini impurity to greedily construct a decision tree for classification.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Classifiers/ClassificationTree.php)

**Interfaces:** [Estimator](#estimators), [Learner](#learner), [Probabilistic](#probabilistic), [Verbose](#verbose), [Persistable](#persistable)

**Compatibility:** Categorical, Continuous

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | max depth | PHP_INT_MAX | int | The maximum depth of a branch. |
| 2 | max leaf size | 3 | int | The max number of samples that a leaf node can contain. |
| 3 | min purity increase | 0. | float | The minimum increase in purity necessary for a node *not* to be post pruned. |
| 4 | max features | Auto | int | The max number of features to consider when determining a best split. |
| 5 | tolerance | 1e-3 | float | A small amount of impurity to tolerate when choosing a best split. |

**Additional Methods:**

Return the feature importances calculated during training indexed by feature column:
```php
public featureImportances() : array
```

Return the height of the tree:
```php
public height() : int
```

Return the balance of the tree:
```php
public balance() : int
```

**Example:**

```php
use Rubix\ML\Classifiers\ClassificationTree;

$estimator = new ClassificationTree(30, 7, 0.1, 4, 1e-4);
```

### Dummy Classifier
A classifier that uses a user-defined [Guessing Strategy](#guessing-strategies) to make predictions. Dummy Classifier is useful to provide a sanity check and to compare performance with an actual classifier.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Classifiers/DummyClassifier.php)

**Interfaces:** [Estimator](#estimators), [Learner](#learner), [Persistable](#persistable)

**Compatibility:** Categorical, Continuous, Resource

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | strategy | Popularity Contest | object | The guessing strategy to employ when guessing the outcome of a sample. |

**Additional Methods:**

This estimator does not have any additional methods.

**Example:**

```php
use Rubix\ML\Classifiers\DummyClassifier;
use Rubix\ML\Other\Strategies\PopularityContest;

$estimator = new DummyClassifier(new PopularityContest());
```

### Extra Tree Classifier
An *Extremely Randomized* Classification Tree - these trees differ from standard [Classification Trees](#classification-tree) in that they choose the best split drawn from a random set determined by *max features*, rather than searching the entire column. Extra Trees work well in ensembles such as [Random Forest](#random-forest) or [AdaBoost](#adaboost) as the *weak learner* or they can be used on their own. The strength of Extra Trees are computational efficiency as well as increasing variance of the prediction (if that is desired).

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Classifiers/ExtraTreeClassifier.php)

**Interfaces:** [Estimator](#estimators), [Learner](#learner), [Probabilistic](#probabilistic), [Verbose](#verbose), [Persistable](#persistable)

**Compatibility:** Categorical, Continuous

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | max depth | PHP_INT_MAX | int | The maximum depth of a branch. |
| 2 | max leaf size | 3 | int | The max number of samples that a leaf node can contain. |
| 3 | min purity increase | 0. | float | The minimum increase in purity necessary for a node *not* to be post pruned. |
| 4 | max features | Auto | int | The max number of features to consider when determining a best split. |
| 5 | tolerance | 1e-3 | float | A small amount of impurity to tolerate when choosing a best split. |

**Additional Methods:**

Return the feature importances calculated during training indexed by feature column:
```php
public featureImportances() : array
```

Return the height of the tree:
```php
public height() : int
```

Return the balance of the tree:
```php
public balance() : int
```

**Example:**

```php
use Rubix\ML\Classifiers\ExtraTreeClassifier;

$estimator = new ExtraTreeClassifier(50, 3, 0.10, 4, 1e-3);
```

**References:**

>- P. Geurts et al. (2005). Extremely Randomized Trees.

### Gaussian Naive Bayes
A variate of the [Naive Bayes](#naive-bayes) algorithm that uses a probability density function (*PDF*) over *continuous* features that are assumed to be normally distributed.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Classifiers/GaussianNB.php)

**Interfaces:** [Estimator](#estimators), [Learner](#learner), [Online](#online), [Probabilistic](#probabilistic), [Persistable](#persistable)

**Compatibility:** Continuous

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | priors | Auto | array | The user-defined class prior probabilities as an associative array with class labels as keys and the prior probabilities as values. |

**Additional Methods:**

Return the class prior probabilities:
```php
public priors() : array
```

Return the running mean of each feature column for each class:
```php
public means() : ?array
```

Return the running variance of each feature column for each class:
```php
public variances() : ?array
```

**Example:**

```php
use Rubix\ML\Classifiers\GaussianNB;

$estimator = new GaussianNB([
	'benign' => 0.9,
	'malignant' => 0.1,
]);
```

**References:**

>- T. F. Chan et al. (1979). Updating Formulae and a Pairwise Algorithm for Computing Sample Variances.

### K-d Neighbors
A fast [K Nearest Neighbors](#k-nearest-neighbors) algorithm that uses a K-d tree to divide the training set into neighborhoods whose max size are controlled by the max leaf size parameter. K-d Neighbors does a binary search to locate the nearest neighborhood and then prunes all neighborhoods whose bounding box is further than the kth nearest neighbor found so far. The main advantage of K-d Neighbors over regular brute force KNN is that it is faster, however it cannot be partially trained.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Classifiers/KDNeighbors.php)

**Interfaces:** [Estimator](#estimators), [Learner](#learner), [Probabilistic](#probabilistic), [Persistable](#persistable)

**Compatibility:** Continuous

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | k | 3 | int | The number of neighboring training samples to consider when making a prediction. |
| 2 | kernel | Euclidean | object | The distance kernel used to compute the distance between sample points. |
| 3 | weighted | true | bool | Should we use the inverse distances as confidence scores when making predictions? |
| 4 | max leaf size | 30 | int | The max number of samples in a leaf node (*neighborhood*). |

**Additional Methods:**

Return the base k-d tree instance:
```php
public tree() : KDTree
```

**Example:**

```php
use Rubix\ML\Classifiers\KDNeighbors;
use Rubix\ML\Kernels\Distance\Euclidean;

$estimator = new KDNeighbors(3, new Euclidean(), false, 10);
```

### K Nearest Neighbors
A distance-based algorithm that locates the K nearest neighbors from the training set and uses a weighted vote to classify the unknown sample.

> **Note**: K Nearest Neighbors is considered a *lazy* learner because it does the majority of its computation at inference. For a fast tree-based version, see [KD Neighbors](#k-d-neighbors).

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Classifiers/KNearestNeighbors.php)

**Interfaces:** [Estimator](#estimators), [Learner](#learner), [Online](#online), [Probabilistic](#probabilistic), [Persistable](#persistable)

**Compatibility:** Continuous

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | k | 3 | int | The number of neighboring training samples to consider when making a prediction. |
| 2 | kernel | Euclidean | object | The distance kernel used to compute the distance between sample points. |
| 3 | weighted | true | bool | Should we use the inverse distances as confidence scores when making predictions? |

**Additional Methods:**

This estimator does not have any additional methods.

**Example:**

```php
use Rubix\ML\Classifiers\KNearestNeighbors;
use Rubix\ML\Kernels\Distance\Manhattan;

$estimator = new KNearestNeighbors(3, new Manhattan(), true);
```

### Logistic Regression
A type of linear classifier that uses the logistic (*sigmoid*) function to estimate the probabilities of exactly *two* classes.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Classifiers/LogisticRegression.php)

**Interfaces:** [Estimator](#estimators), [Learner](#learner), [Online](#online), [Probabilistic](#probabilistic), [Verbose](#verbose), [Persistable](#persistable)

**Compatibility:** Continuous

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | batch size | 100 | int | The number of training samples to process at a time. |
| 2 | optimizer | Adam | object | The gradient descent optimizer used to train the underlying network. |
| 3 | alpha | 1e-4 | float | The amount of L2 regularization to apply to the weights of the network. |
| 4 | epochs | 1000 | int | The maximum number of training epochs to execute. |
| 5 | min change | 1e-4 | float | The minimum change in the cost function necessary to continue training. |
| 6 | cost fn | Cross Entropy | object | The function that computes the cost of an erroneous activation during training. |

**Additional Methods:**

Return the average loss of a sample at each epoch of training:
```php
public steps() : array
```

Return the underlying neural network instance or *null* if untrained:
```php
public network() : Network|null
```

**Example:**

```php
use Rubix\ML\Classifers\LogisticRegression;
use Rubix\ML\NeuralNet\Optimizers\Adam;
use Rubix\ML\NeuralNet\CostFunctions\CrossEntropy;

$estimator = new LogisticRegression(10, new Adam(0.001), 1e-4, 100, 1e-4, new CrossEntropy());
```

### Multi Layer Perceptron
A multiclass feedforward [Neural Network](#neural-network) classifier that uses a series of user-defined [hidden layers](#hidden) as intermediate computational units. Multiple layers and non-linear activation functions allow the Multi Layer Perceptron to handle complex deep learning problems.

> **Note**: The MLP features progress monitoring which stops training early if it can no longer make progress. It also utilizes snapshotting to make sure that it always has the best settings of the model parameters even if progress began to decline during training.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Classifiers/MultiLayerPerceptron.php)

**Interfaces:** [Estimator](#estimators), [Learner](#learner), [Online](#online), [Probabilistic](#probabilistic), [Verbose](#verbose), [Persistable](#persistable)

**Compatibility:** Continuous

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | hidden | | array | An array composing the hidden layers of the neural network. |
| 2 | batch size | 100 | int | The number of training samples to process at a time. |
| 3 | optimizer | Adam | object | The gradient descent optimizer used to train the underlying network. |
| 4 | alpha | 1e-4 | float | The amount of L2 regularization to apply to the weights of the network. |
| 5 | epochs | 1000 | int | The maximum number of training epochs to execute. |
| 6 | min change | 1e-4 | float | The minimum change in the cost function necessary to continue training. |
| 7 | cost fn | Cross Entropy | object | The function that computes the cost of an erroneous activation during training. |
| 8 | holdout | 0.1 | float | The ratio of samples to hold out for progress monitoring. |
| 9 | metric | FBeta | object | The validation metric used to monitor the training progress of the network. |
| 10 | window | 3 | int | The number of epochs to consider when determining if the algorithm should terminate or keep training. |

**Additional Methods:**

Return the average loss of a sample at each epoch of training:
```php
public steps() : array
```

Return the validation scores at each epoch of training:
```php
public scores() : array
```

Returns the underlying neural network instance or *null* if untrained:
```php
public network() : Network|null
```

**Example:**

```php
use Rubix\ML\Classifiers\MultiLayerPerceptron;
use Rubix\ML\NeuralNet\Layers\Dense;
use Rubix\ML\NeuralNet\Layers\Dropout;
use Rubix\ML\NeuralNet\Layers\Activation;
use Rubix\ML\NeuralNet\ActivationFunctions\LeakyReLU;
use Rubix\ML\NeuralNet\ActivationFunctions\PReLU;
use Rubix\ML\NeuralNet\Optimizers\Adam;
use Rubix\ML\NeuralNet\CostFunctions\CrossEntropy;
use Rubix\ML\CrossValidation\Metrics\MCC;

$estimator = new MultiLayerPerceptron([
    new Dense(100),
	new Activation(new LeakyReLU()),
	new Dropout(0.3),
    new Dense(100),
	new Activation(new LeakyReLU()),
	new Dropout(0.3),
	new Dense(50),
	new Activation(new LeakyReLU()),
	new Dense(30),
    new PReLU(),
    new Dense(10),
	new PReLU(),
], 100, new Adam(0.001), 1e-4, 1000, 1e-3, new CrossEntropy(), 0.1, new MCC(), 3);
```

**References:**

>- G. E. Hinton. (1989). Connectionist learning procedures.

### Naive Bayes
Probability-based classifier that estimates posterior probabilities of each class using Bayes' Theorem and the conditional probabilities calculated during training. The *naive* part relates to the fact that the algorithm assumes that all features are independent (non-correlated).

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Classifiers/NaiveBayes.php)

**Interfaces:** [Estimator](#estimators), [Learner](#learner), [Online](#online), [Probabilistic](#probabilistic), [Persistable](#persistable)

**Compatibility:** Categorical

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | alpha | 1.0 | float | The amount of additive (Laplace/Lidstone) smoothing to apply to the probabilities. |
| 2 | priors | Auto | array | The class prior probabilities as an associative array with class labels as keys and the prior probabilities as values. |

**Additional Methods:**

Return the class prior probabilities:
```php
public priors() : array
```

Return the negative log probabilities of each feature given each class label:
```php
public probabilities() : array
```

**Example:**

```php
use Rubix\ML\Classifiers\NaiveBayes;

$estimator = new NaiveBayes(2.5, [
	'spam' => 0.3,
	'not spam' => 0.7,
]);
```

### Radius Neighbors
Radius Neighbors is a spatial tree-based classifier that takes the weighted vote of each neighbor within a fixed user-defined radius measured by a kernel distance function.

> **Note**: Unknown samples with 0 samples from the training set that are within radius will be labeled as outliers (*-1*).

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Classifiers/RadiusNeighbors.php)

**Interfaces:** [Estimator](#estimators), [Learner](#learner), [Probabilistic](#probabilistic), [Persistable](#persistable)

**Compatibility:** Continuous

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | radius | 1.0 | float | The radius within which points are considered neighboors. |
| 2 | kernel | Euclidean | object | The distance kernel used to compute the distance between sample points. |
| 3 | weighted | true | bool | Should we use the inverse distances as confidence scores when making predictions? |
| 4 | max leaf size | 30 | int | The max number of samples in a leaf node (*ball*). |

**Additional Methods:**

Return the base ball tree instance:
```php
public tree() : BallTree
```

**Example:**

```php
use Rubix\ML\Classifiers\RadiusNeighbors;
use Rubix\ML\Kernels\Distance\Manhattan;

$estimator = new RadiusNeighbors(50.0, new Manhattan(), false, 30);
```

### Random Forest
Ensemble classifier that trains Decision Trees ([Classification Trees](#classification-tree) or [Extra Trees](#extra-tree)) on a random subset (*bootstrap* set) of the training data. A prediction is made based on the probability scores returned from each tree in the forest averaged and weighted equally.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Classifiers/RandomForest.php)

**Interfaces:** [Estimator](#estimators), [Learner](#learner), [Probabilistic](#probabilistic), [Parallel](#parallel), [Persistable](#persistable)

**Compatibility:** Categorical, Continuous

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | base | ClassificationTree | object | The base tree estimator. |
| 2 | estimators | 100 | int | The number of estimators to train in the ensemble. |
| 3 | ratio | 0.1 | float | The ratio of random samples to train each estimator with. |

**Additional Methods:**

This estimator does not have any additional methods.

**Example:**

```php
use Rubix\ML\Classifiers\RandomForest;
use Rubix\ML\Classifiers\ClassificationTree;

$estimator = new RandomForest(new ClassificationTree(10), 400, 0.1);
```

**References:**

>- L. Breiman. (2001). Random Forests.
>- L. Breiman et al. (2005). Extremely Randomized Trees.

### Softmax Classifier
A generalization of [Logistic Regression](#logistic-regression) for multiclass problems using a single layer neural network with a Softmax output layer.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Classifiers/SoftmaxClassifier.php)

**Interfaces:** [Estimator](#estimators), [Learner](#learner), [Online](#online), [Probabilistic](#probabilistic), [Verbose](#verbose), [Persistable](#persistable)

**Compatibility:** Continous

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | batch size | 100 | int | The number of training samples to process at a time. |
| 2 | optimizer | Adam | object | The gradient descent optimizer used to train the underlying network. |
| 3 | alpha | 1e-4 | float | The amount of L2 regularization to apply to the weights of the network. |
| 4 | epochs | 1000 | int | The maximum number of training epochs to execute. |
| 5 | min change | 1e-4 | float | The minimum change in the cost function necessary to continue training. |
| 6 | cost fn | Cross Entropy | object | The function that computes the cost of an erroneous activation during training. |

**Additional Methods:**

Return the average loss of a sample at each epoch of training:
```php
public steps() : array
```

Return the underlying neural network instance or *null* if untrained:
```php
public network() : Network|null
```

**Example:**

```php
use Rubix\ML\Classifiers\SoftmaxClassifier;
use Rubix\ML\NeuralNet\Optimizers\Momentum;
use Rubix\ML\NeuralNet\CostFunctions\CrossEntropy;

$estimator = new SoftmaxClassifier(256, new Momentum(0.001), 1e-4, 300, 1e-4, new CrossEntropy());
```

### SVC
The multiclass Support Vector Machine (SVM) Classifier is a maximum margin classifier that can efficiently perform non-linear classification by implicitly mapping feature vectors into high dimensional feature space (called the *kernel trick*).

> **Note**: This estimator requires the [SVM PHP extension](https://php.net/manual/en/book.svm.php) which uses the LIBSVM engine written in C++ under the hood.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Classifiers/SVC.php)

**Interfaces:** [Estimator](#estimators), [Learner](#learner), [Persistable](#persistable)

**Compatibility:** Continous

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | c | 1.0 | float | The parameter that defines the width of the margin used to separate the classes. |
| 2 | kernel | RBF | object | The kernel function used to operate in higher dimensions. |
| 3 | shrinking | true | bool | Should we use the shrinking heuristic? |
| 4 | tolerance | 1e-3 | float | The minimum change in the cost function necessary to continue training. |
| 5 | cache size | 100. | float | The size of the kernel cache in MB. |

**Additional Methods:**

This estimator does not have any additional methods.

**Example:**

```php
use Rubix\ML\Classifiers\SVC;
use Rubix\ML\Kernels\SVM\Linear;

$estimator = new SVC(1.0, new Linear(), true, 1e-3, 100.);
```

**References:**

>- C. Chang et al. (2011). LIBSVM: A library for support vector machines.

---
### Clusterers
Clustering is a technique in machine learning that focuses on grouping samples in such a way that the groups are similar. Another way of looking at it is that clusterers take unlabeled data points and assign them a label (cluster number).

### DBSCAN
*Density-Based Spatial Clustering of Applications with Noise* is a clustering algorithm able to find non-linearly separable and arbitrarily-shaped clusters. In addition, DBSCAN also has the ability to mark outliers as *noise* and thus can be used as a *quasi* [Anomaly Detector](#anomaly-detectors).

> **Note**: Noise samples are assigned the cluster number *-1*.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Clusterers/DBSCAN.php)

**Interfaces:** [Estimator](#estimators)

**Compatibility:** Continuous

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | radius | 0.5 | float | The maximum distance between two points to be considered neighbors. |
| 2 | min density | 5 | int | The minimum number of points within radius of each other to form a cluster. |
| 3 | kernel | Euclidean | object | The distance kernel used to compute the distance between sample points. |
| 4 | max leaf size | 30 | int | The max number of samples in a leaf node (*ball*). |

> **Note**: The smaller the radius, the tighter the clusters will be.

**Additional Methods:**

This estimator does not have any additional methods.

**Example:**

```php
use Rubix\ML\Clusterers\DBSCAN;
use Rubix\ML\Kernels\Distance\Diagonal;

$estimator = new DBSCAN(4.0, 5, new Diagonal(), 20);
```

**References:**

>- M. Ester et al. (1996). A Densty-Based Algorithm for Discovering Clusters.

### Fuzzy C Means
Distance-based soft clusterer that allows samples to belong to multiple clusters if they fall within a *fuzzy* region controlled by the *fuzz* parameter.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Clusterers/FuzzyCMeans.php)

**Interfaces:** [Estimator](#estimators), [Learner](#learner), [Probabilistic](#probabilistic), [Verbose](#verbose), [Persistable](#persistable)

**Compatibility:** Continuous

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | c | | int | The number of target clusters. |
| 2 | fuzz | 2.0 | float | Determines the bandwidth of the fuzzy area. |
| 3 | kernel | Euclidean | object | The distance kernel used to compute the distance between sample points. |
| 4 | epochs | 300 | int | The maximum number of training rounds to execute. |
| 5 | min change | 10. | float | The minimum change in the inertia for the algorithm to continue training. |
| 6 | seeder | PlusPlus | object | The seeder used to initialize the cluster centroids. |

**Additional Methods:**

Return the *c* computed centroids of the training set:
```php
public centroids() : array
```

Returns the inertia at each epoch from  the last round of training:
```php
public steps() : array
```

**Example:**

```php
use Rubix\ML\Clusterers\FuzzyCMeans;
use Rubix\ML\Kernels\Distance\Euclidean;
use Rubix\ML\Clusterers\Seeders\Random;

$estimator = new FuzzyCMeans(5, 1.2, new Euclidean(), 400, 1., new Random());
```

**References:**

>- J. C. Bezdek et al. (1984). FCM: The Fuzzy C-Means Clustering Algorithm.

### Gaussian Mixture
A Gaussian Mixture model (GMM) is a probabilistic model for representing the presence of clusters within an overall population without requiring a sample to know which sub-population it belongs to a priori. GMMs are similar to centroid-based clusterers like [K Means](#k-means) but allow both the centers (*means*) *and* the radii (*variances*) to be learned as well.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Clusterers/GaussianMixture.php)

**Interfaces:** [Estimator](#estimators), [Learner](#learner), [Probabilistic](#probabilistic), [Verbose](#verbose), [Persistable](#persistable)

**Compatibility:** Continuous

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | k | | int | The number of target clusters. |
| 2 | epochs | 100 | int | The maximum number of training rounds to execute. |
| 3 | min change | 1e-3 | float | The minimum change in the components necessary for the algorithm to continue training. |
| 6 | seeder | PlusPlus | object | The seeder used to initialize the Guassian components. |

**Additional Methods:**

Return the cluster prior probabilities based on their representation over all training samples:
```php
public priors() : array
```

Return the running means of each feature column for each cluster:
```php
public means() : array
```

Return the variance of each feature column for each cluster:
```php
public variances() : array
```

**Example:**

```php
use Rubix\ML\Clusterers\GaussianMixture;
use Rubix\ML\Clusterers\Seeders\KMC2;

$estimator = new GaussianMixture(5, 1e-4, 100, new KMC2(50));
```

**References:**

>- A. P. Dempster et al. (1977). Maximum Likelihood from Incomplete Data via the EM Algorithm.
>- J. Blomer et al. (2016). Simple Methods for Initializing the EM Algorithm for Gaussian Mixture Models.

### K Means
A fast online centroid-based hard clustering algorithm capable of clustering linearly separable data points given some prior knowledge of the target number of clusters (defined by *k*). K Means with inertia is trained using adaptive mini batch gradient descent and minimizes the inertial cost function. Inertia is defined as the sum of the distances between each sample and its nearest cluster centroid.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Clusterers/KMeans.php)

**Interfaces:** [Estimator](#estimators), [Learner](#learner), [Online](#online), [Probabilistic](#probabilistic), [Persistable](#persistable), [Verbose](#verbose)

**Compatibility:** Continuous

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | k | | int | The number of target clusters. |
| 2 | batch size | 100 | int | The size of each mini batch in samples. |
| 3 | kernel | Euclidean | object | The distance kernel used to compute the distance between sample points. |
| 4 | epochs | 300 | int | The maximum number of training rounds to execute. |
| 5 | min change | 10. | float | The minimum change in the inertia for training to continue. |
| 6 | seeder | PlusPlus | object | The seeder used to initialize the cluster centroids. |

**Additional Methods:**

Return the *k* computed centroids of the training set:
```php
public centroids() : array
```

Return the number of training samples that each centroid is responsible for:
```php
public sizes() : array
```

Return the value of the inertial function at each epoch from the last round of training:
```php
public steps() : array
```

**Example:**

```php
use Rubix\ML\Clusterers\KMeans;
use Rubix\ML\Clusterers\Seeders\PlusPlus;
use Rubix\ML\Kernels\Distance\Euclidean;

$estimator = new KMeans(3, 100, new Euclidean(), 300, 10., new PlusPlus());
```

**References:**

>- D. Sculley. (2010). Web-Scale K-Means Clustering.

### Mean Shift
A hierarchical clustering algorithm that uses peak finding to locate the local maxima (*centroids*) of a training set given by a radius constraint.

> **Note**: Seeding Mean Shift with a [Seeder](#seeders) can speed up convergence using large datasets. The default is to initialize all training samples as seeds.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Clusterers/MeanShift.php)

**Interfaces:** [Estimator](#estimators), [Learner](#learner), [Probabilistic](#probabilistic), [Verbose](#verbose), [Persistable](#persistable)

**Compatibility:** Continuous

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | radius | | float | The bandwidth of the radial basis function. |
| 2 | kernel | Euclidean | object | The distance kernel used to compute the distance between samples. |
| 3 | max leaf size | 30 | int | The max number of samples in a leaf node (*ball*). |
| 4 | epochs | 100 | int | The maximum number of training rounds to execute. |
| 5 | min change | 1e-4 | float | The minimum change in centroids necessary for the algorithm to continue training. |
| 6 | seeder | None | object | The seeder used to initialize the cluster centroids. |
| 7 | ratio | 0.2 | float | The ratio of samples from the training set to seed the algorithm with. |

**Additional Methods:**

Estimate the radius of a cluster that encompasses a certain percentage of the total training samples:
```php
public static estimateRadius(Dataset $dataset, float $percentile = 30., ?Distance $distance = null) : float
```

> **Note**: Since radius estimation scales quadratically in the number of samples, for large datasets you can speed up the process by running it on a sample subset of the training data.

Return the centroids computed from the training set:
```php
public centroids() : array
```

Returns the amount of centroid shift during each epoch of training:
```php
public steps() : array
```

**Example:**

```php
use Rubix\ML\Clusterers\MeanShift;
use Rubix\ML\Kernels\Distance\Diagonal;
use Rubix\ML\Clusterers\Seeders\KMC2;

$radius = MeanShift::estimateRadius($dataset, 30., new Diagonal()); // Automatically choose radius

$estimator = new MeanShift($radius, new Diagonal(), 30, 2000, 1e-6, new KMC2(), 0.1);
```

**References:**

>- M. A. Carreira-Perpinan et al. (2015). A Review of Mean-shift Algorithms for Clustering.
>- D. Comaniciu et al. (2012). Mean Shift: A Robust Approach Toward Feature Space Analysis.

### Seeders
Seeders are responsible for initializing k starting clusters used by certain learners such as [KMeans](#k-means), [Mean Shift](#mean-shift), and [Gaussian Mixture](#gaussian-mixture).

### K-MC2
This is a fast [Plus Plus](#plus-plus) approximator that replaces the brute force method with a substantially faster Markov Chain Monte Carlo (MCMC) sampling method with comparable performance.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Clusterers/Seeders/KMC2.php)

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | m | 50 | int | The number of candidate nodes in the Markov Chain. |
| 2 | kernel | Euclidean | object | The distance kernel used to compute the distance between samples. |

**Example:**

```php
use Rubix\ML\Clusterers\Seeders\KMC2;
use Rubix\ML\Kernels\Distance\Euclidean;

$seeder = new KMC2(200, new Euclidean());
```

**References:**

>- O. Bachem et al. (2016). Approximate K-Means++ in Sublinear Time.

### Plus Plus
This seeder attempts to maximize the likelihood of seeding distant clusters while still remaining random. It does so by sequentially selecting random samples weighted by their distance from the previous seed.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Clusterers/Seeders/PlusPlus.php)

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | kernel | Euclidean | object | The distance kernel used to compute the distance between samples. |

**Example:**

```php
use Rubix\ML\Clusterers\Seeders\PlusPlus;
use Rubix\ML\Kernels\Distance\Minkowski;

$seeder = new PlusPlus(new Minkowski(5.0));
```

**References:**

>- D. Arthur et al. (2006). k-means++: The Advantages of Careful Seeding.
>- A. Stetco et al. (2015). Fuzzy C-means++: Fuzzy C-means with effective seeding initialization.

### Random
Completely random selection of k seeds from the given dataset.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Clusterers/Seeders/Random.php)

**Parameters:**

This seeder does not have any parameters.

**Example:**

```php
use Rubix\ML\Clusterers\Seeders\Random;

$seeder = new Random();
```

---
### Embedders
Manifold learning is a type of non-linear dimensionality reduction used primarily for visualizing high dimensional datasets in low (1 to 3) dimensions. Embedders are manifold learners that provide the `predict()` API for embedding a dataset. The predictions of an Embedder are the low dimensional embeddings as n-dimensional arrays where n is the dimensionality of the embedding.

### t-SNE
*T-distributed Stochastic Neighbor Embedding* is a two-stage non-linear manifold learning algorithm based on batch Gradient Descent. During the first stage (*early* stage) the samples are exaggerated to encourage distant clusters. Since the t-SNE cost function (KL Divergence) has a rough gradient, momentum is employed to help escape bad local minima.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Embedders/TSNE.php)

**Interfaces:** [Estimator](#estimators), [Verbose](#verbose)

**Compatibility:** Continous

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | dimensions | 2 | int | The number of dimensions of the target embedding. |
| 2 | perplexity | 30 | int | The number of effective nearest neighbors to refer to when computing the variance of the Gaussian over that sample. |
| 3 | exaggeration | 12. | float | The factor to exaggerate the distances between samples during the early stage of fitting. |
| 4 | rate | 100. | float | The learning rate that controls the step size. |
| 5 | kernel | Euclidean | object | The distance kernel to use when measuring distances between samples. |
| 6 | epochs | 1000 | int | The number of times to iterate over the embedding. |
| 7 | min gradient | 1e-8 | float | The minimum gradient necessary to continue embedding. |
| 8 | window | 3 | int | The number of most recent epochs to consider when determining an early stop. |

**Additional Methods:**

Return the magnitudes of the gradient at each epoch from the last embedding:
```php
public steps() : array
```

**Example:**

```php
use Rubi\ML\Embedders\TSNE;
use Rubix\ML\Kernels\Manhattan;

$embedder = new TSNE(2, 30, 12., 10., new Manhattan(), 500, 1e-6, 5);
```

**References:**

>- L. van der Maaten et al. (2008). Visualizing Data using t-SNE.
>- L. van der Maaten. (2009). Learning a Parametric Embedding by Preserving Local Structure.

---
### Regressors
Regressors predict continuous (*real-valued*) outcomes. In contrast to a [Classifier](#classifiers), which predicts one of *k* discrete outcomes, regressors output a single numerical value which has the potential to range from -∞ to ∞ but will often fall within the range of the training labels.

### Adaline
Adaptive Linear Neuron or (*Adaline*) is a type of single layer [neural network](#neural-network) with a linear output neuron. Training is equivalent to solving [Ridge](#ridge) regression iteratively using mini batch Gradient Descent.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Regressors/Adaline.php)

**Interfaces:** [Estimator](#estimators), [Learner](#learner), [Online](#online), [Verbose](#verbose), [Persistable](#persistable)

**Compatibility:** Continuous

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | batch size | 100 | int | The number of training samples to process at a time. |
| 2 | optimizer | Adam | object | The gradient descent optimizer used to train the underlying network. |
| 3 | alpha | 1e-4 | float | The amount of L2 regularization to apply to the weights of the network. |
| 4 | epochs | 100 | int | The maximum number of training epochs to execute. |
| 5 | min change | 1e-4 | float | The minimum change in the cost function necessary to continue training. |
| 6 | cost fn | Least Squares | object | The function that computes the cost of an erroneous activation during training. |

**Additional Methods:**

Return the average loss of a sample at each epoch of training:
```php
public steps() : array
```

Return the underlying neural network instance or *null* if untrained:
```php
public network() : Network|null
```

**Example:**

```php
use Rubix\ML\Classifers\Adaline;
use Rubix\ML\NeuralNet\Optimizers\Adam;
use Rubix\ML\NeuralNet\CostFunctions\HuberLoss;

$estimator = new Adaline(10, new Adam(0.001), 500, 1e-6, new HuberLoss(2.5));
```

### Dummy Regressor
Regressor that guesses output values based on a user-defined [Guessing Strategy](#guessing-strategies). Dummy Regressor is useful to provide a sanity check and to compare performance against actual Regressors.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Regressors/DummyRegressor.php)

**Interfaces:** [Estimator](#estimators), [Learner](#learner), [Persistable](#persistable)

**Compatibility:** Categorical, Continuous, Resource

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | strategy | Mean | object | The guessing strategy to employ when guessing the outcome of a sample. |

**Additional Methods:**

This estimator does not have any additional methods.

**Example:**

```php
use Rubix\ML\Regressors\DummyRegressor;
use Rubix\ML\Other\Strategies\BlurryPercentile;

$estimator = new DummyRegressor(new BlurryPercentile(56.5, 0.1));
```

### Extra Tree Regressor
An *Extremely Randomized* Regression Tree, these trees differ from standard [Regression Trees](#regression-tree) in that they choose a split drawn from a random set determined by the max features parameter, rather than searching the entire column for the best split.

> **Note**: Decision tree based algorithms can handle both categorical and continuous features at the same time.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Regressors/ExtraTreeRegressor.php)

**Interfaces:** [Estimator](#estimators), [Learner](#learner), [Verbose](#verbose), [Persistable](#persistable)

**Compatibility:** Categorical, Continuous

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | max depth | PHP_INT_MAX | int | The maximum depth of a branch that is allowed. |
| 2 | max leaf size | 3 | int | The max number of samples that a leaf node can contain. |
| 3 | min purity increase | 0. | float | The minimum increase in purity necessary for a node *not* to be post pruned. |
| 4 | max features | Auto | int | The number of features to consider when determining a best split. |
| 5 | tolerance | 1e-4 | float | A small amount of impurity to tolerate when choosing a best split. |

**Additional Methods:**

Return the feature importances calculated during training indexed by feature column:
```php
public featureImportances() : array
```

Return the height of the tree:
```php
public height() : int
```

Return the balance of the tree:
```php
public balance() : int
```

**Example:**

```php
use Rubix\ML\Classifiers\ExtraTreeRegressor;

$estimator = new ExtraTreeRegressor(30, 3, 0.05, 20, 1e-4);
```

**References:**

>- P. Geurts et al. (2005). Extremely Randomized Trees.

### Gradient Boost
Gradient Boost is a stage-wise additive model that uses a Gradient Descent boosting paradigm for training  boosters (Regression Trees) to correct the error residuals of a *weak* base learner.

> **Note**: The default base regressor is a Dummy Regressor using the *Mean* Strategy and the default booster is a Regression Tree with a max depth of 3.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Regressors/GradientBoost.php)

**Interfaces:** [Estimator](#estimators), [Learner](#learner), [Ensemble](#ensemble), [Verbose](#verbose), [Persistable](#persistable)

**Compatibility:** Depends on base learner

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | booster | Regression Tree | object | The regressor that will fix up the error residuals of the base learner. |
| 2 | rate | 0.1 | float | The learning rate of the ensemble. |
| 3 | estimators | 100 | int | The number of estimators to train in the ensemble. |
| 4 | ratio | 0.8 | float | The ratio of samples to subsample from the training dataset per epoch. |
| 5 | min change | 1e-4 | float | The minimum change in the cost function necessary to continue training. |
| 6 | tolerance | 1e-3 | float | The amount of mean squared error to tolerate before early stopping. |
| 7 | base | Dummy Regressor | object | The *weak* learner to be boosted. |

**Additional Methods:**

Return the training error at each epoch:
```php
public steps() : array
```

**Example:**

```php
use Rubix\ML\Regressors\GradientBoost;
use Rubix\ML\Regressors\DummyRegressor;
use Rubix\ML\Regressors\RegressionTree;
use Rubix\ML\Other\Strategies\Mean;

$estimator = new GradientBoost(new RegressionTree(3), 0.1, 400, 0.3, 1e-4, 1e-3, new DummyRegressor(new Mean()));
```

**References:**

>- J. H. Friedman. (2001). Greedy Function Approximation: A Gradient Boosting Machine.

### K-d Neighbors Regressor
A fast implementation of [KNN Regressor](#knn-regressor) using a spatially-aware K-d tree. The KDN Regressor works by locating the neighborhood of a sample via binary search and then does a brute force search only on the samples close to or within the neighborhood. The main advantage of K-d Neighbors over brute force KNN is inference speed, however you no longer have the ability to partially train.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Regressors/KDNeighborsRegressor.php)

**Interfaces:** [Estimator](#estimators), [Learner](#learner), [Persistable](#persistable)

**Compatibility:** Continuous

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | k | 3 | int | The number of neighboring training samples to consider when making a prediction. |
| 2 | kernel | Euclidean | object | The distance kernel used to compute the distance between sample points. |
| 3 | weighted | true | bool | Should we use the inverse distances as confidence scores when making predictions? |
| 4 | max leaf size | 30 | int | The max number of samples in a leaf node (*neighborhood*). |

**Additional Methods:**

Return the base k-d tree instance:
```php
public tree() : KDTree
```

**Example:**

```php
use Rubix\ML\Regressors\KDNeighborsRegressor;
use Rubix\ML\Kernels\Distance\Minkowski;

$estimator = new KDNeighborsRegressor(5, new Minkowski(4.0), true, 30);
```

### KNN Regressor
A version of [K Nearest Neighbors](#knn-regressor) that uses the average (mean) outcome of K nearest data points to make continuous valued predictions suitable for regression problems.

> **Note**: K Nearest Neighbors is considered a *lazy* learning estimator because it does the majority of its computation at prediction time.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Regressors/KNNRegressor.php)

**Interfaces:** [Estimator](#estimators), [Learner](#learner), [Online](#online), [Persistable](#persistable)

**Compatibility:** Continuous

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | k | 3 | int | The number of neighboring training samples to consider when making a prediction. |
| 2 | kernel | Euclidean | object | The distance kernel used to compute the distance between sample points. |
| 3 | weighted | true | bool | Should we use the inverse distances as confidence scores when making predictions? |

**Additional Methods:**

This estimator does not have any additional methods.

**Example:**

```php
use Rubix\ML\Regressors\KNNRegressor;
use Rubix\ML\Kernels\Distance\Minkowski;

$estimator = new KNNRegressor(2, new Minkowski(3.0), false);
```

### MLP Regressor
A multi layer feedforward [Neural Network](#neural-network) with a continuous output layer suitable for regression problems. Like the [Multi Layer Perceptron](#multi-layer-perceptron) classifier, the MLP Regressor is able to tackle deep learning problems by forming higher-order representations of the features using intermediate computational units called *hidden* layers.

> **Note**: The MLP features progress monitoring which stops training early if it can no longer make progress. It also utilizes snapshotting to make sure that it always has the best settings of the model parameters even if progress began to decline during training.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Regressors/MLPRegressor.php)

**Interfaces:** [Estimator](#estimators), [Learner](#learner), [Online](#online), [Verbose](#verbose), [Persistable](#persistable)

**Compatibility:** Continuous

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | hidden | | array | An array composing the hidden layers of the neural network. |
| 2 | batch size | 100 | int | The number of training samples to process at a time. |
| 3 | optimizer | Adam | object | The gradient descent optimizer used to train the underlying network. |
| 4 | alpha | 1e-4 | float | The amount of L2 regularization to apply to the weights of the network. |
| 5 | epochs | 1000 | int | The maximum number of training epochs to execute. |
| 6 | min change | 1e-4 | float | The minimum change in the cost function necessary to continue training. |
| 7 | cost fn | Least Squares | object | The function that computes the cost of an erroneous activation during training. |
| 8 | holdout | 0.1 | float | The ratio of samples to hold out for progress monitoring. |
| 9 | metric | Mean Squared Error | object | The validation metric used to monitor the training progress of the network. |
| 10 | window | 3 | int | The number of epochs to consider when determining if the algorithm should terminate or keep training. |

**Additional Methods:**

Return the average loss of a sample at each epoch of training:
```php
public steps() : array
```

Return the validation scores at each epoch of training:
```php
public scores() : array
```

Returns the underlying neural network instance or *null* if untrained:
```php
public network() : Network|null
```

**Example:**

```php
use Rubix\ML\Regressors\MLPRegressor;
use Rubix\ML\NeuralNet\Layers\Dense;
use Rubix\ML\NeuralNet\Layers\Activation;
use Rubix\ML\NeuralNet\ActivationFunctions\LeakyReLU;
use Rubix\ML\NeuralNet\Optimizers\RMSProp;
use Rubix\ML\CrossValidation\Metrics\RSquared;

$estimator = new MLPRegressor([
	new Dense(50),
	new Activation(new LeakyReLU(0.1)),
	new Dense(50),
	new Activation(new LeakyReLU(0.1)),
	new Dense(50),
	new Activation(new LeakyReLU(0.1)),
], 256, new RMSProp(0.001), 1e-3, 100, 1e-5, new LeastSquares(), 0.1, new RSquared(), 3);
```

**References:**

>- G. E. Hinton. (1989). Connectionist learning procedures.

### Radius Neighbors Regressor
This is the regressor version of [Radius Neighbors](#radius-neighbors) classifier implementing a binary spatial tree under the hood for fast radius queries. The prediction is a weighted average of each label from the training set that is within a fixed user-defined radius.

> **Note**: Unknown samples with 0 samples from the training set that are within radius will be labeled *NaN*.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Regressors/RadiusNeighborsRegressor.php)

**Interfaces:** [Estimator](#estimators), [Learner](#learner), [Persistable](#persistable)

**Compatibility:** Continuous

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | radius | 1.0 | float | The radius within which points are considered neighboors. |
| 2 | kernel | Euclidean | object | The distance kernel used to compute the distance between sample points. |
| 3 | weighted | true | bool | Should we use the inverse distances as confidence scores when making predictions? |
| 4 | max leaf size | 30 | int | The max number of samples in a leaf node (*ball*). |

**Additional Methods:**

Return the height of the tree:
```php
public height() : int
```

Return the balance of the tree:
```php
public balance() : int
```

**Example:**

```php
use Rubix\ML\Regressors\RadiusNeighborsRegressor;
use Rubix\ML\Kernels\Distance\Diagonal;

$estimator = new RadiusNeighborsRegressor(0.5, new Diagonal(), true, 20);
```

### Regression Tree
A Decision Tree learning algorithm (CART) that performs greedy splitting by minimizing the impurity (variance) of the labels at each decision node split.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Regressors/RegressionTree.php)

**Interfaces:** [Estimator](#estimators), [Learner](#learner), [Verbose](#verbose), [Persistable](#persistable)

**Compatibility:** Categorical, Continuous

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | max depth | PHP_INT_MAX | int | The maximum depth of a branch. |
| 2 | max leaf size | 3 | int | The maximum number of samples that a leaf node can contain. |
| 3 | min purity increase | 0. | float | The minimum increase in purity necessary for a node *not* to be post pruned. |
| 4 | max features | Auto | int | The maximum number of features to consider when determining a best split. |
| 5 | tolerance | 1e-4 | float | A small amount of impurity to tolerate when choosing a best split. |

**Additional Methods:**

Return the feature importances calculated during training indexed by feature column:
```php
public featureImportances() : array
```

Return the height of the tree:
```php
public height() : int
```

Return the balance of the tree:
```php
public balance() : int
```

**Example:**

```php
use Rubix\ML\Regressors\RegressionTree;

$estimator = new RegressionTree(30, 2, 35., null, 1e-4);
```

### Ridge
L2 penalized least squares linear regression solved using closed-form equation.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Regressors/Ridge.php)

**Interfaces:** [Estimator](#estimators), [Learner](#learner), [Persistable](#persistable)

**Compatibility:** Continuous

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | alpha | 1.0 | float | The L2 regularization penalty amount to be added to the weight coefficients. |

**Additional Methods:**

Return the weights of the model:
```php
public weights() : array|null
```

Return the bias parameter:
```php
public bias() : float|null
```

**Example:**
```php
use Rubix\ML\Regressors\Ridge;

$estimator = new Ridge(2.0);
```

### SVR
The Support Vector Machine Regressor is a maximum margin algorithm for the purposes of regression. Similarly to the [Support Vector Machine Classifier](#svc), the model produced by SVR (*R* for regression) depends only on a subset of the training data, because the cost function for building the model ignores any training data close to the model prediction given by parameter *epsilon*. Thus, the value of epsilon defines a margin of tolerance where no penalty is given to errors.

> **Note**: This estimator requires the [SVM PHP extension](https://php.net/manual/en/book.svm.php) which uses the LIBSVM engine written in C++ under the hood.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Regressors/SVR.php)

**Interfaces:** [Estimator](#estimators), [Learner](#learner), [Persistable](#persistable)

**Compatibility:** Continous

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | c | 1.0 | float | The parameter that defines the width of the margin used to separate the classes. |
| 2 | epsilon | 0.1 | float | Specifies the margin within which no penalty is associated in the training loss. |
| 3 | kernel | RBF | object | The kernel function used to operate in higher dimensions. |
| 4 | shrinking | true | bool | Should we use the shrinking heuristic? |
| 5 | tolerance | 1e-3 | float | The minimum change in the cost function necessary to continue training. |
| 6 | cache size | 100. | float | The size of the kernel cache in MB. |

**Additional Methods:**

This estimator does not have any additional methods.

**Example:**

```php
use Rubix\ML\Classifiers\SVC;
use Rubix\ML\Kernels\SVM\Linear;

$estimator = new SVR(1.0, 0.03, new RBF(), true, 1e-3, 256.);
```

**References**

>- C. Chang et al. (2011). LIBSVM: A library for support vector machines.
>- A. Smola et al. (2003). A Tutorial on Support Vector Regression.

---
### Meta-Estimators
Meta-estimators enhance base estimators by adding additional functionality such as [data preprocessing](#pipeline), [model persistence](#persistent-model), and [model averaging](#bootstrap-aggregator). Meta-estimators take on the type (*Classifier*, *Regressor*, etc.) of the base estimator they wrap and allow methods on the base estimator to be called from the parent.

### Bootstrap Aggregator
Bootstrap Aggregating (or *bagging* for short) is a model averaging technique designed to improve the stability and performance of a user-specified base estimator by training a number of them on a unique *bootstrapped* training set sampled at random with replacement. 

> **Note**: Bootstrap Aggregator is not compatible with clusterers or embedders.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/BootstrapAggregator.php)

**Interfaces:** [Estimator](#estimators), [Learner](#learner), [Parallel](#parallel), [Persistable](#persistable)

**Compatibility:** Depends on base learner

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | base | | object | The base estimator to be used in the ensemble. |
| 2 | estimators | 10 | int | The number of base estimators to train in the ensemble. |
| 3 | ratio | 0.5 | float | The ratio of samples from the training set to train each base estimator with. |

**Additional Methods:**

This meta estimator does not have any additional methods.

**Example:**

```php
use Rubix\ML\BootstrapAggregator;
use Rubix\ML\Regressors\RegressionTree;

$estimator = new BootstrapAggregator(new RegressionTree(20), 300, 0.2);
```

**References:**

>- L. Breiman. (1996). Bagging Predictors.

### Committee Machine
A voting ensemble that aggregates the predictions of a committee of heterogeneous learners (referred to as *experts*). The committee employs a user-specified influence-based scheme to make final predictions.

> **Note**: Influence values can be arbitrary as they are normalized upon instantiation.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/CommitteeMachine.php)

**Interfaces:** [Estimator](#estimators), [Learner](#learner), [Parallel](#parallel), [Verbose](#verbose), [Persistable](#persistable)

**Compatibility:** Depends on the base learners

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | experts | | array | An array of learner instances that will comprise the committee. |
| 2 | influences | Equal | array | The influence score for each expert in the committee. |

**Additional Methods:**

Return the learner instances of the committee:
```php
public experts() : array
```

Return the normalized influence scores of each expert in the committee:
```php
public influences() : array
```

**Example:**

```php
use Rubix\ML\CommitteeMachine;
use Rubix\ML\Classifiers\GaussianNB;
use Rubix\ML\Classifiers\RandomForest;
use Rubix\ML\Classifiers\ClassificationTree;
use Rubix\ML\Classifiers\KDNeighbors;
use Rubix\ML\Classifiers\SoftmaxClassifier;
use Rubix\ML\NeuralNet\Optimizers\Momentum;

$estimator = new CommitteeMachine([
    new GaussianNB(),
    new RandomForest(new ClassificationTree(4), 100, 0.3),
    new KDNeighbors(3),
    new SoftmaxClassifier(100, new Mometum(0.001)),
], [4, 6, 5, 4]);
```

**References:**

>- [1] H. Drucker. (1997). Fast Committee Machines for Regression and Classification.

### Grid Search
Grid Search is an algorithm that optimizes hyper-parameter selection. From the user's perspective, the process of training and predicting is the same, however, under the hood, Grid Search trains one estimator per combination of parameters and the best model is selected as the base estimator.

> **Note**: You can choose the parameters to search manually or you can generate them randomly or in a grid using the [Params](#params) helper.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/GridSearch.php)

**Interfaces:** [Estimator](#estimators), [Learner](#learner), [Parallel](#parallel), [Persistable](#persistable), [Verbose](#verbose)

**Compatibility:** Depends on the base learner

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | base | | string | The fully qualified class name of the base Estimator. |
| 2 | grid | | array | An array of [n-tuples](#what-is-a-tuple) where each tuple contains possible parameters for a given constructor location by ordinal. |
| 3 | metric | Auto | object | The validation metric used to score each set of hyper-parameters. |
| 4 | validator | KFold | object | An instance of a validator object (HoldOut, KFold, etc.) that will be used to test each model. |

**Additional Methods:**

Return an array of every possible parameter combination:
```php
public combinations() : array
```

A [tuple](#what-is-a-tuple) containing the best parameters and their validation score:
```php
public best() : array
```

Return the underlying base estimator:
```php
public estimator() : Estimator
```

**Example:**

```php
use Rubix\ML\GridSearch;
use Rubix\ML\Classifiers\KNearestNeighbors;
use Rubix\ML\Kernels\Distance\Euclidean;
use Rubix\ML\Kernels\Distance\Manhattan;
use Rubix\ML\CrossValidation\Metrics\FBeta;
use Rubix\ML\CrossValidation\KFold;

$grid = [
	[1, 3, 5, 10], [new Euclidean(), new Manhattan()], [true, false],
];

$estimator = new GridSearch(KNearestNeightbors::class, $grid, new FBeta(), new KFold(10));
```

### Persistent Model
The Persistent Model wrapper gives the estimator two additional methods (`save()` and `load()`) that allow the estimator to be saved and retrieved from storage.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/PersistentModel.php)

**Interfaces:** [Estimator](#estimators), [Learner](#learner), [Probabilistic](#probabilistic), [Verbose](#verbose)

**Compatibility:** Depends on the base learner

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | base | | object | An instance of the base estimator to be persisted. |
| 2 | persister | | object | The persister object used to store the model data. |

**Additional Methods:**

Save the persistent model to storage:
```php
public save() : void
```

Load the persistent model from storage given a persister:
```php
public static load(Persister $persister) : self
```

**Example:**

```php
use Rubix\ML\PersistentModel;
use Rubix\ML\Classifiers\RandomForest;
use Rubix\ML\Persisters\Filesystem;
use Rubix\ML\Persisters\Serializers\Native;

$persister = new Filesystem('random_forest.model', true, new Native());

$estimator = new PersistentModel(new RandomForest(100), $persister);
```

### Pipeline
Pipeline is a meta estimator responsible for transforming the input data by applying a series of [transformer](#transformers) middleware. Pipeline accepts a base estimator and a list of transformers to apply to the input data before it is fed to the estimator. Under the hood, Pipeline will automatically fit the training set upon training and transform any [Dataset object](#dataset-objects) supplied as an argument to one of the base Estimator's methods, including `train()` and `predict()`. With the *elastic* mode enabled, Pipeline can update the fitting of certain transformers during online (*partial*) training.

> **Note**: Since transformations are applied to dataset objects in place (without making a copy), using the dataset in a program after it has been run through Pipeline may have unexpected results. If you need a *clean* dataset object to call multiple methods with, you can use the PHP clone syntax to keep an original (untransformed) copy in memory.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Pipeline.php)

**Interfaces:** [Estimator](#estimators), [Learner](#learner), [Online](#online), [Probabilistic](#probabilistic), [Persistable](#persistable), [Verbose](#verbose)

**Compatibility:** Depends on base learner and transformers

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | transformers |  | array | The transformer middleware to be applied to the input data in order. |
| 2 | estimator |  | object | An instance of the base estimator to receive transformed data. |
| 3 | elastic | true | bool | Should we update the elastic transformers during partial training? |

**Additional Methods:**

This meta estimator does not have any additional methods.

**Example:**

```php
use Rubix\ML\Pipeline;
use Rubix\ML\Classifiers\SoftmaxClassifier;
use Rubix\ML\NeuralNet\Optimizer\Adam;
use Rubix\ML\Transformers\NumericStringConverter;
use Rubix\ML\Transformers\MissingDataImputer;
use Rubix\ML\Transformers\OneHotEncoder;
use Rubix\ML\Transformers\PrincipalComponentAnalysis;

$estimator = new Pipeline([
    new NumericStringConverter(),
	new MissingDataImputer('?'),
	new OneHotEncoder(), 
	new PrincipalComponentAnalysis(20),
], new SoftmaxClassifier(100, new Adam(0.001)));
```

---
### Backends
A computation Backend is responsible for executing a queue of Deferred computations, often in parallel. They are used by objects that implement the [Parallel](#parallel) interface to carry out their underlying computations.

### Amp
[Amp](https://amphp.org/) Parallel is a multiprocessing subsystem that requires no extensions. It uses a non-blocking concurrency framework that implements coroutines using PHP generator functions under the hood.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Backends/Amp.php)

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | workers | 4 | int | The maximum number of processes to execute in parallel. |

**Additional Methods:**

Build an Amp backend based on autodetected core count:
```php
public static autotune() : self
```

**Example:**

```php
use Rubix\ML\Backends\Amp;

$backend = Amp::autotune();

$backend = new Amp(16);
```

### Serial
The Serial backend executes tasks sequentially inside of a single PHP process. The advantage of the Serial backend is that it has zero overhead, thus it may be faster than a parallel backend in cases where the computions are minimal such as with small datasets.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Backends/Serial.php)

**Parameters:**

This backend does not have any additional parameters.

**Additional Methods:**

This backend does not have any additional methods.

**Example:**

```php
use Rubix\ML\Backends\Serial;

$backend = new Serial();
```

---
### Persisters
Persisters are responsible for persisting a *Persistable* object to storage and are also used by the [Persistent Model](#persistent-model) meta-estimator to save and restore models.

To store a persistable estimator:
```php
public save(Persistable $persistable) : void
```

Load the saved model from persistence:
```php
public load() : Persistable
```

### Filesystem
Filesystems are local or remote storage drives that are organized by files and folders. The Filesystem persister saves models to a file at a given path and automatically keeps backups of the latest versions of your models.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Persisters/Filesystem.php)

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | path | | string | The path to the model file on the filesystem. |
| 2 | history | false | bool | Should we keep a history of past saves? |
| 3 | serializer | Native | object | The serializer used to convert to and from storage format. |

**Additional Methods:**

This persister does not have any additional methods.

**Example:**

```php
use Rubix\ML\Persisters\Filesystem;
use Rubix\ML\Persisters\Serializers\Binary;

$persister = new Filesystem('/path/to/example.model', true, new Binary());
```

### Redis DB
Redis is a high performance in-memory key value store that can be used to persist your trained models over a network.

> **Note**: The Redis persister requires the PHP [Redis extension](https://github.com/phpredis/phpredis) and a properly configured Redis server.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Persisters/RedisDB.php)

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | key | | string | The key of the object in the database. |
| 2 | host | '127.0.0.1' | string | The hostname or IP address of the Redis server. |
| 3 | port | 6379 | int | The port of the Redis server. |
| 4 | db | 0 | int | The database number. |
| 5 | password | None | string | An optional password to access a password-protected server. |
| 6 | serializer | Native | object | The serializer used to convert to and from storage format. |
| 7 | timeout | 2.5 | float | The time in seconds to wait for a response from the server before timing out. |

**Additional Methods:**

Return an associative array of info from the Redis server:
```php
public info() : array
```

**Example:**

```php
use Rubix\ML\Persisters\RedisDB;
use Rubix\ML\Persisters\Serializers\Native;

$persister = new RedisDB('model:sentiment', '127.0.0.1', 6379, 2, 'secret', new Native(), 2.5);
```

### Serializers
Serializers take Persistable estimators and convert them between their in-memory and storage representations.

To serialize a Persistable:
```php
public serialize(Persistable $persistable) : string
```

To unserialize a Persistable:
```php
public unserialize(string $data) : Persistable
```

### Binary Serializer
Converts Persistable objects to and from binary format.

> **Note**: The [Igbinary extension](https://github.com/igbinary/igbinary) is needed to use this serializer.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Persisters/Serializers/Binary.php)

**Parameters:**

This serializer does not have any parameters.

**Example:**

```php
use Rubix\ML\Persisters\Serializers\Binary;

$serializer = new Binary();
```

### Native
The native plain text serialization format that is bundled with PHP core.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Persisters/Serializers/Native.php)

**Parameters:**

This serializer does not have any parameters.

**Example:**

```php
use Rubix\ML\Persisters\Serializers\Native;

$serializer = new Native();
```

---
### Transformers
Transformers take [Dataset](#dataset-objects) objects and apply blanket transformations to the features contained within them. They are often used as part of a [Pipeline](#pipeline) or they can be used by themselves. Examples of transformations are scaling and centering, normalization, dimensionality reduction, missing data imputation, and feature selection.

The transformer directly transforms the samples in place via the `transform()` method:
```php
public transform(array &$samples) : void
```

> **Note**: To transform a dataset without having to pass the raw sample, instead you can pass a transformer object to the `apply()` method on a Dataset object.

### Stateful
For stateful transformers, the `fit()` method will allow the transformer to compute any necessary information from the training set in order to carry out its future transformations. You can think of *fitting* a transformer like *training* a learner.

To fit the transformer to a training set:
```php
public fit(Dataset $dataset) : void
```

Check if the transformer has been fitted:
```php
public fitted() : bool
```

**Example**

```php
use Rubix\ML\Transformers\OneHotEncoder;

$transformer = new OneHotEncoder();

$transformer->fit($dataset);
```

### Elastic
Some transformers are able to adapt to new training data. The `update()` method on transformers that implement the Elastic interface can be used to modify the fitting of the transformer with new data even after it has previously been fitted. *Updating* is the transformer equivalent to *partially training* an online learner.

```php
public update(Dataset $dataset) : void
```

**Example**

```php
use Rubix\ML\Transformers\ZScaleStandardizer;

$transformer = new ZScaleStandardizer();

$folds = $dataset->fold(3);

$transformer->fit($folds[0]);

$transformer->update($folds[1]);

$transformer->update($folds[2]);
```

### Dense Random Projector
The Dense Random Projector uses a random matrix sampled from a dense uniform distribution [-1, 1] to reduce the dimensionality of a dataset by projecting it onto a vector space of target dimensionality.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Transformers/DenseRandomProjector.php)

**Interfaces:** [Transformer](#transformers), [Stateful](#stateful)

**Compatibility:** Continuous only

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | dimensions | | int | The number of target dimensions to project onto. |

**Additional Methods:**

Estimate the minimum dimensionality needed given total sample size and *max distortion* using the Johnson-Lindenstrauss lemma:
```php
public static estimate(int $n, float $maxDistortion = 0.1) : int
```

**Example:**
```php
use Rubix\ML\Transformers\DenseRandomProjector;

$transformer = new DenseRandomProjector(50);
```

**References:**

>- D. Achlioptas. (2003). Database-friendly random projections: Johnson-Lindenstrauss with binary coins.

### Gaussian Random Projector
A random projector is a dimensionality reducer based on the Johnson-Lindenstrauss lemma that uses a random matrix to project feature vectors onto a user-specified number of dimensions. It is faster than most non-randomized dimensionality reduction techniques such as [PCA](#principal-component-analysis) or [LDA](#linear-discriminant-analysis) and it offers similar results. This version utilizes a random matrix sampled from a smooth Gaussian distribution.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Transformers/GaussianRandomProjector.php)

**Interfaces:** [Transformer](#transformers), [Stateful](#stateful)

**Compatibility:** Continuous only

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | dimensions | | int | The number of target dimensions to project onto. |

**Additional Methods:**

Estimate the minimum dimensionality needed given total sample size and *max distortion* using the Johnson-Lindenstrauss lemma:
```php
public static estimate(int $n, float $maxDistortion = 0.1) : int
```

**Example:**

```php
use Rubix\ML\Transformers\GaussianRandomProjector;

$transformer = new GaussianRandomProjector(100);
```

### HTML Stripper
Removes any HTML tags that may be in the text of a categorical variable.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Transformers/HTMLStripper.php)

**Interfaces:** [Transformer](#transformers)

**Compatibility:** Categorical

**Parameters:**

This transformer does not have any parameters.

**Additional Methods:**

This transformer does not have any additional methods.

**Example:**

```php
use Rubix\ML\Transformers\HTMLStripper;

$transformer = new HTMLStripper();
```

### Image Resizer
The Image Resizer scales and crops images to a user specified width, height, and color depth.

> **Note**: Note that the [GD extension](https://php.net/manual/en/book.image.php) is required to use this transformer.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Transformers/ImageResizer.php)

**Interfaces:** [Transformer](#transformers)

**Compatibility:** Resource (GD Image)

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | width | 32 | int | The width of the transformed image. |
| 2 | heights | 32 | int | The height of the transformed image. |
| 3 | driver | 'gd' | string | The PHP extension to use for image processing ('gd' *or* 'imagick'). |

**Additional Methods:**

This transformer does not have any additional methods.

**Example:**

```php
use Rubix\ML\Transformers\ImageResizer;

$transformer = new ImageResizer(28, 28, 'gd');
```

### Image Vectorizer
Image Vectorizer takes images and converts them into a flat vector of raw color channel data.

> **Note**: Note that the [GD extension](https://php.net/manual/en/book.image.php) is required to use this transformer.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Transformers/ImageVectorizer.php)

**Interfaces:** [Transformer](#transformers)

**Compatibility:** Resource (Images)

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | channels | 3 | int | The channel depth i.e the number of rgba channels to encode starting with red. |

**Additional Methods:**

This transformer does not have any additional methods.

**Example:**

```php
use Rubix\ML\Transformers\ImageVectorizer;

$transformer = new ImageVectorizer(3);
```

### Interval Discretizer
This transformer creates an equi-width histogram for each continuous feature column and encodes a discrete category with an automatic bin label. The Interval Discretizer is helpful when converting continuous features to categorical features so they can be learned by an estimator that supports categorical features natively.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Transformers/IntervalDiscretizer.php)

**Interfaces:** [Transformer](#transformers), [Stateful](#stateful)

**Compatibility:** Continuous

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | bins | 5 | int | The number of bins (discrete features) per continuous feature column. |

**Additional Methods:**

Return the possible categories of each feature column:
```php
public categories() : array
```

Return the intervals of each continuous feature column calculated during fitting:
```php
public intervals() : array
```

**Example:**

```php
use Rubix\ML\Transformers\IntervalDiscretizer;

$transformer = new IntervalDiscretizer(10);
```

### L1 Normalizer
Transform each sample vector in the sample matrix such that each feature is divided by the L1 norm (or *magnitude*) of that vector.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Transformers/L1Normalizer.php)

**Interfaces:** [Transformer](#transformers)

**Compatibility:** Continuous only

**Parameters:**

This transformer does not have any parameters.

**Additional Methods:**

This transformer does not have any additional methods.

**Example:**

```php
use Rubix\ML\Transformers\L1Normalizer;

$transformer = new L1Normalizer();
```

### L2 Normalizer
Transform each sample vector in the sample matrix such that each feature is divided by the L2 norm (or *magnitude*) of that vector.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Transformers/L2Normalizer.php)

**Interfaces:** [Transformer](#transformers)

**Compatibility:** Continuous only

**Parameters:**

This transformer does not have any parameters.

**Additional Methods:**

This transformer does not have any additional methods.

**Example:**

```php
use Rubix\ML\Transformers\L2Normalizer;

$transformer = new L2Normalizer();
```

### Lambda Function
Run a stateless lambda function (*anonymous* function) over the sample matrix. The lambda function receives the sample matrix (and labels if applicable) as an argument and should return the transformed sample matrix and labels in a [2-tuple](#what-is-a-tuple).

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Transformers/LambdaFunction.php)

**Interfaces:** [Transformer](#transformers)

**Compatibility** Depends on user function

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | lambda | | callable | The lambda function to run over the sample matrix. |

**Additional Methods:**

This transformer does not have any additional methods.

**Example:**

```php
use Rubix\ML\Transformers\LambdaFunction;

$transformer = new LambdaFunction(function ($samples, $labels) {
	$samples = array_map(function ($sample) {
		return [array_sum($sample)];
	}, $samples);

	return [$samples, $labels];
});
```

### Linear Discriminant Analysis
A supervised dimensionality reduction technique that selects the most discriminating features based on class labels. In other words, LDA finds a linear combination of features that characterizes or best separates two or more classes.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Transformers/LinearDiscriminantAnalysis.php)

**Interfaces:** [Transformer](#transformers), [Stateful](#stateful)

**Compatibility:** Continuous only

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | dimensions | | int | The target number of dimensions to project onto. |

**Additional Methods:**

Return the amount of variance that has been preserved by the transformation:
```php
public explainedVar() : ?float
```

Return the amount of variance lost by discarding the noise components:
```php
public noiseVar() : ?float
```

Return the percentage of information lost due to the transformation:
```php
public lossiness() : ?float
```

**Example:**

```php
use Rubix\ML\Transformers\LinearDiscriminantAnalysis;

$transformer = new LinearDiscriminantAnalysis(20);
```

### Max Absolute Scaler
Scale the sample matrix by the maximum absolute value of each feature column independently such that the feature will be between -1 and 1.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Transformers/MaxAbsoluteScaler.php)

**Interfaces:** [Transformer](#transformers), [Stateful](#stateful), [Elastic](#elastic)

**Compatibility:** Continuous

**Parameters:**

This transformer does not have any parameters.

**Additional Methods:**

Return the maximum absolute values for each feature column:
```php
public maxabs() : array
```

**Example:**

```php
use Rubix\ML\Transformers\MaxAbsoluteScaler;

$transformer = new MaxAbsoluteScaler();
```

### Min Max Normalizer
The *Min Max* Normalizer scales the input features to a value between a user-specified range (*default* 0 to 1).

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Transformers/MinMaxNormalizer.php)

**Interfaces:** [Transformer](#transformers), [Stateful](#stateful), [Elastic](#elastic)

**Compatibility:** Continuous

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | min | 0. | float | The minimum value of the transformed features. |
| 2 | max | 1. | float | The maximum value of the transformed features. |

**Additional Methods:**

Return the minimum values for each fitted feature column:
```php
public minimums() : ?array
```

Return the maximum values for each fitted feature column:
```php
public maximums() : ?array
```

**Example:**

```php
use Rubix\ML\Transformers\MinMaxNormalizer;

$transformer = new MinMaxNormalizer(-5., 5.);
```

### Missing Data Imputer
In the real world, it is common to have data with missing values here and there. The Missing Data Imputer replaces missing value *placeholder* values with a guess based on a given [Strategy](#guessing-strategies).

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Transformers/MissingDataImputer.php)

**Interfaces:** [Transformer](#transformers), [Stateful](#stateful)

**Compatibility:** Categorical, Continuous

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | placeholder | '?' | string or numeric | The placeholder value that denotes a missing value. |
| 2 | continuous strategy | Mean | object | The guessing strategy to employ for continuous feature columns. |
| 3 | categorical strategy | K Most Frequent | object | The guessing strategy to employ for categorical feature columns. |

**Additional Methods:**

This transformer does not have any additional methods.

**Example:**

```php
use Rubix\ML\Transformers\MissingDataImputer;
use Rubix\ML\Other\Strategies\BlurryPercentile;
use Rubix\ML\Other\Strategies\PopularityContest;

$transformer = new MissingDataImputer('?', new BlurryPercentile(0.61), new PopularityContest());
```

### Numeric String Converter
Convert all numeric strings into their integer and floating point countertypes. Useful for when extracting from a source that only recognizes data as string types.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Transformers/NumericStringConverter.php)

**Interfaces:** [Transformer](#transformers)

**Compatibility:** Categorical

**Parameters:**

This transformer does not have any parameters.

**Additional Methods:**

This transformer does not have any additional methods.

**Example:**

```php
use Rubix\ML\Transformers\NumericStringConverter;

$transformer = new NumericStringConverter();
```

### One Hot Encoder
The One Hot Encoder takes a column of categorical features and produces a n-d *one-hot* representation where n is equal to the number of unique categories in that column. A 0 in any location indicates that a category represented by that column is not present whereas a 1 indicates that a category is present in the sample.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Transformers/OneHotEncoder.php)

**Interfaces:** [Transformer](#transformers), [Stateful](#stateful)

**Compatibility:** Categorical

**Parameters:**

This transformer does not have any parameters.

**Additional Methods:**

This transformer does not have any additional methods.

**Example:**

```php
use Rubix\ML\Transformers\OneHotEncoder;

$transformer = new OneHotEncoder();
```

### Polynomial Expander
This transformer will generate polynomials up to and including the specified *degree* of each continuous feature column. Polynomial expansion is sometimes used to fit data that is non-linear using a linear estimator such as [Ridge](#ridge) or [Logistic Regression](#logistic-regression).

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Transformers/PolynomialExpander.php)

**Interfaces:** [Transformer](#transformers)

**Compatibility:** Continuous only

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | degree | 2 | int | The highest degree polynomial to generate from each feature vector. |

**Additional Methods:**

This transformer does not have any additional methods.

**Example:**

```php
use Rubix\ML\Transformers\PolynomialExpander;

$transformer = new PolynomialExpander(3);
```

### Principal Component Analysis
Principal Component Analysis or *PCA* is a dimensionality reduction technique that aims to transform the feature space by the k *principal components* that explain the most variance of the data where *k* is the dimensionality of the output specified by the user. PCA is used to compress high dimensional samples down to lower dimensions such that they would retain as much of the information as possible.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Transformers/PrincipalComponentAnalysis.php)

**Interfaces:** [Transformer](#transformers), [Stateful](#stateful)

**Compatibility:** Continuous only

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | dimensions | None | int | The target number of dimensions to project onto. |

**Additional Methods:**

Return the amount of variance that has been preserved by the transformation:
```php
public explainedVar() : ?float
```

Return the amount of variance lost by discarding the noise components:
```php
public noiseVar() : ?float
```

Return the percentage of information lost due to the transformation:
```php
public lossiness() : ?float
```

**Example:**

```php
use Rubix\ML\Transformers\PrincipalComponentAnalysis;

$transformer = new PrincipalComponentAnalysis(15);
```

**References:**

>- H. Abdi et al. (2010). Principal Component Analysis.

### Robust Standardizer
This standardizer transforms continuous features by centering them around the median and scaling by the median absolute deviation (*MAD*). The use of robust statistics make this standardizer more immune to outliers than the [Z Scale Standardizer](#z-scale-standardizer) which used mean and variance.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Transformers/RobustStandardizer.php)

**Interfaces:** [Transformer](#transformers), [Stateful](#stateful)

**Compatibility:** Continuous

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | center | true | bool | Should we center the sample dataset? |

**Additional Methods:**

Return the medians calculated by fitting the training set:
```php
public medians() : array
```

Return the median absolute deviations calculated during fitting:
```php
public mads() : array
```

**Example:**

```php
use Rubix\ML\Transformers\RobustStandardizer;

$transformer = new RobustStandardizer(true);
```

### Sparse Random Projector
The Sparse Random Projector uses a random matrix sampled from a sparse Gaussian distribution (mostly *0*s) to reduce the dimensionality of a dataset.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Transformers/SparseRandomProjector.php)

**Interfaces:** [Transformer](#transformers), [Stateful](#stateful)

**Compatibility:** Continuous only

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | dimensions | | int | The number of target dimensions to project onto. |

**Additional Methods:**

Calculate the minimum dimensionality needed given total sample size and max distortion using the Johnson-Lindenstrauss lemma:
```php
public static minDimensions(int $n, float $maxDistortion = 0.1) : int
```

**Example:**

```php
use Rubix\ML\Transformers\SparseRandomProjector;

$transformer = new SparseRandomProjector(30);
```

**References:**

>- D. Achlioptas. (2003). Database-friendly random projections: Johnson-Lindenstrauss with binary coins.

### Stop Word Filter
Removes user-specified words from any categorical feature column including blobs of text.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Transformers/StopWordFilter.php)

**Interfaces:** [Transformer](#transformers)

**Compatibility:** Categorical

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | stop words | | array | A list of stop words to filter out of each text feature. |

**Additional Methods:**

This transformer does not have any additional methods.

```php
use Rubix\ML\Transformers\StopWordFilter;

$transformer = new StopWordFilter(['i', 'me', 'my', ...]);
```

### Text Normalizer
This transformer converts all text to lowercase and *optionally* removes extra whitespace.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Transformers/TextNormalizer.php)

**Interfaces:** [Transformer](#transformers)

**Compatibility:** Categorical

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | trim | false | bool | Should we trim excess whitespace? |

**Additional Methods:**

This transformer does not have any additional methods.

```php
use Rubix\ML\Transformers\TextNormalizer;

$transformer = new TextNormalizer(true);
```

### TF-IDF Transformer
*Term Frequency - Inverse Document Frequency* is a measure of how important a word is to a document. The TF-IDF value increases proportionally with the number of times a word appears in a document (*TF*) and is offset by the frequency of the word in the corpus (*IDF*).

> **Note**: This transformer assumes that its input is made up of word frequency vectors such as those created by the [Word Count Vectorizer](#word-count-vectorizer).

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Transformers/TfIdfTransformer.php)

**Interfaces:** [Transformer](#transformers), [Stateful](#stateful), [Elastic](#elastic)

**Compatibility:** Continuous only

**Parameters:**
This transformer does not have any parameters.

**Additional Methods:**

Return the inverse document frequencies calculated during fitting:
```php
public idfs() : ?array
```

**Example:**

```php
use Rubix\ML\Transformers\TfIdfTransformer;

$transformer = new TfIdfTransformer();
```

**References:**

>- S. Robertson. (2003). Understanding Inverse Document Frequency: On theoretical arguments for IDF.

### Variance Threshold Filter
A type of feature selector that selects feature columns that have a greater variance than the user-specified threshold.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Transformers/VarianceThresholdFilter.php)

**Interfaces:** [Transformer](#transformers), [Stateful](#stateful)

**Compatibility:** Continuous

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | threshold | 0. | float | Feature columns with a variance greater than this threshold will be selected. |

**Additional Methods:**

Return the columns that were selected during fitting:
```php
public selected() : array
```

**Example:**

```php
use Rubix\ML\Transformers\VarianceThresholdFilter;

$transformer = new VarianceThresholdFilter(50);
```

### Word Count Vectorizer
The Word Count Vectorizer builds a vocabulary from the training samples and transforms text blobs into fixed length feature vectors. Each feature column represents a word or *token* from the vocabulary and the value denotes the number of times that word appears in a given sample.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Transformers/WordCountVectorizer.php)

**Interfaces:** [Transformer](#transformers), [Stateful](#stateful)

**Compatibility:** Categorical

**Parameters:**
| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | max vocabulary | PHP_INT_MAX | int | The maximum number of words to encode into each document vector. |
| 2 | min document frequency | 1 | int | The minimum number of documents a word must appear in to be added to the vocabulary. |
| 3 | tokenizer | Word | object | The tokenizer that extracts individual words from samples of text. |

**Additional Methods:**

Return the fitted vocabulary i.e. the words that will be vectorized:
```php
public vocabulary() : array
```

Return the size of the vocabulary:
```php
public size() : int
```

**Example:**

```php
use Rubix\ML\Transformers\WordCountVectorizer;
use Rubix\ML\Other\Tokenizers\SkipGram;

$transformer = new WordCountVectorizer(10000, 3, new SkipGram());
```

### Z Scale Standardizer
A method of centering and scaling a dataset such that it has 0 mean and unit variance, also known as a Z Score.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Transformers/ZScaleStandardizer.php)

**Interfaces:** [Transformer](#transformers), [Stateful](#stateful), [Elastic](#elastic)

**Compatibility:** Continuous

**Parameters:**
| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | center | true | bool | Should we center the sample dataset? |

**Additional Methods:**

Return the means calculated by fitting the training set:
```php
public means() : array
```

Return the variances calculated during fitting:
```php
public variances() : array
```

Return the standard deviations calculated during fitting:
```php
public stddevs() : array
```

**Example:**

```php
use Rubix\ML\Transformers\ZScaleStandardizer;

$transformer = new ZScaleStandardizer(true);
```

**References:**

>- T. F. Chan et al. (1979). Updating Formulae and a Pairwise Algorithm for Computing Sample Variances.

---
### Neural Network
A number of estimators in Rubix are implemented as a Neural Network under the hood. Neural nets are trained using an iterative supervised learning process called Gradient Descent with Backpropagation that repeatedly takes small steps towards minimizing a user-defined cost function. Networks can have an arbitrary number of intermediate computational layers called *hidden* layers. Hidden layers can perform a number of different functions including higher order feature detection, non-linear activation, normalization, and regularization.

### Activation Functions
The input to a node in the network is often passed through an Activation Function (sometimes referred to as a *transfer* function) which determines its output behavior. In the context of a *biologically inspired* neural network, the activation function is an abstraction representing the rate of action potential firing of a neuron.

Activation Functions can be broken down into three classes - Sigmoidal (or *S* shaped) such as [Hyperbolic Tangent](#hyperbolic-tangent), Rectifiers such as [ELU](#elu) and LeakyReLU(#leaky-relu), and Radial Basis Functions (*RBFs*) such as [Gaussian](#gaussian).

### ELU
*Exponential Linear Units* are a type of rectifier that soften the transition from non-activated to activated using the exponential function.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/NeuralNet/ActivationFunctions/ELU.php)

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | alpha | 1.0 | float | The value at which leakage will begin to saturate. Ex. alpha = 1.0 means that the output will never be less than -1.0 when inactivated. |

**Example:**

```php
use Rubix\ML\NeuralNet\ActivationFunctions\ELU;

$activationFunction = new ELU(5.0);
```

**References:**

>- D. A. Clevert et al. (2016). Fast and Accurate Deep Network Learning by Exponential Linear Units.

### Hyperbolic Tangent
S-shaped function that squeezes the input value into an output space between -1 and 1 centered at 0.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/NeuralNet/ActivationFunctions/HyperbolicTangent.php)

**Parameters:**

This activation Function does not have any parameters.

**Example:**

```php
use Rubix\ML\NeuralNet\ActivationFunctions\HyperbolicTangent;

$activationFunction = new HyperbolicTangent();
```

### Leaky ReLU
Leaky Rectified Linear Units are activation functions that output x when x > 0 or a small leakage value determined as the input times the leakage coefficient when x < 0. The amount of leakage is controlled by the *leakage* parameter.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/NeuralNet/ActivationFunctions/LeakyReLU.php)

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | leakage | 0.1 | float | The amount of leakage as a proportion of the input value to allow to pass through when not inactivated. |

**Example:**

```php
use Rubix\ML\NeuralNet\ActivationFunctions\LeakyReLU;

$activationFunction = new LeakyReLU(0.3);
```

**References:**

>- A. L. Maas et al. (2013). Rectifier Nonlinearities Improve Neural Network Acoustic Models.

### ReLU
A Thresholded ReLU (Rectified Linear Unit) only outputs the signal above a user-defined threshold parameter.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/NeuralNet/ActivationFunctions/ReLU.php)

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | threshold | 0. | float | The input value necessary to trigger an activation. |

**Example:**

```php
use Rubix\ML\NeuralNet\ActivationFunctions\ReLU;

$activationFunction = new ReLU(0.1);
```

**References:**

>- A. L. Maas et al. (2013). Rectifier Nonlinearities Improve Neural Network Acoustic Models.
>- K. Konda et al. (2015). Zero-bias Autoencoders and the Benefits of Co-adapting Features.

### SELU
Scaled Exponential Linear Unit is a *self-normalizing* activation function based on the [ELU](#elu) activation function. Neuronal activations of SELU networks automatically converge toward zero mean and unit variance, unlike explicitly normalized networks such as those with [Batch Norm](#batch-norm).

> [Source](https://github.com/RubixML/RubixML/blob/master/src/NeuralNet/ActivationFunctions/SELU.php)

**Parameters:**

This actvation function does not have any parameters.

**Example:**

```php
use Rubix\ML\NeuralNet\ActivationFunctions\SELU;

$activationFunction = new SELU();
```

**References:**

>- G. Klambauer et al. (2017). Self-Normalizing Neural Networks.

### Sigmoid
A bounded S-shaped function (specifically the *Logistic* function) with an output value between 0 and 1.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/NeuralNet/ActivationFunctions/Sigmoid.php)

**Parameters:**

This activation Function does not have any parameters.

**Example:**

```php
use Rubix\ML\NeuralNet\ActivationFunctions\Sigmoid;

$activationFunction = new Sigmoid();
```

### Softmax
The Softmax function is a generalization of the [Sigmoid](#sigmoid) function that *squashes* each activation between 0 and 1 *and* all activations together add up to exactly 1.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/NeuralNet/ActivationFunctions/Softmax.php)

**Parameters:**

This activation function does not have any parameters.

**Example:**

```php
use Rubix\ML\NeuralNet\ActivationFunctions\Softmax;

$activationFunction = new Softmax();
```

### Soft Plus
A smooth approximation of the ReLU function whose output is constrained to be positive.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/NeuralNet/ActivationFunctions/SoftPlus.php)

**Parameters:**

This activation function does not have any parameters.

**Example:**

```php
use Rubix\ML\NeuralNet\ActivationFunctions\SoftPlus;

$activationFunction = new SoftPlus();
```

**References:**

>- X. Glorot et al. (2011). Deep Sparse Rectifier Neural Networks.

### Softsign
A function that squashes the input smoothly between -1 and 1.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/NeuralNet/ActivationFunctions/Softsign.php)

**Parameters:**

This activation function does not have any parameters.

**Example:**

```php
use Rubix\ML\NeuralNet\ActivationFunctions\Softsign;

$activationFunction = new Softsign();
```

**References:**

>- X. Glorot et al. (2010). Understanding the Difficulty of Training Deep Feedforward Neural Networks

### Cost Functions
In neural networks, the cost function is a function that the network tries to minimize during training. The cost of a particular activation is defined as the difference between the output of the network and what the correct output should be given the ground truth label. Different cost functions have different ways of punishing erroneous activations and thus produce differently shaped gradients when backpropagated.

### Cross Entropy
Cross Entropy, or *log loss*, measures the performance of a classification model whose output is a probability value between 0 and 1. Cross-entropy loss increases as the predicted probability diverges from the actual label. So predicting a probability of .012 when the actual observation label is 1 would be bad and result in a high loss value. A perfect score would have a log loss of 0.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/NeuralNet/CostFunctions/CrossEntropy.php)

**Parameters:**

This cost function does not have any parameters.

**Example:**

```php
use Rubix\ML\NeuralNet\CostFunctions\CrossEntropy;

$costFunction = new CrossEntropy();
```

### Huber Loss
The *pseudo* Huber Loss function transitions between L1 and L2 (Least Squares) loss at a given pivot point (*delta*) such that the function becomes more quadratic as the loss decreases. The combination of L1 and L2 loss makes Huber Loss robust to outliers while maintaining smoothness near the minimum.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/NeuralNet/CostFunctions/HuberLoss.php)

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | delta | 1. | float | The pivot point i.e the point where numbers larger will be evaluated with an L1 loss while number smaller will be evaluated with an L2 loss. |

**Example:**

```php
use Rubix\ML\NeuralNet\CostFunctions\HuberLoss;

$costFunction = new HuberLoss(0.5);
```

### Least Squares
Least Squares or *quadratic* loss is a function that measures the squared error between the target output and the actual output of the network.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/NeuralNet/CostFunctions/LeastSquares.php)

**Parameters:**

This cost function does not have any parameters.

**Example:**

```php
use Rubix\ML\NeuralNet\CostFunctions\LeastSquares;

$costFunction = new LeastSquares();
```

### Relative Entropy
Relative Entropy or *Kullback-Leibler divergence* is a measure of how the expectation and activation of the network diverge.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/NeuralNet/CostFunctions/RelativeEntropy.php)

**Parameters:**

This cost function does not have any parameters.

**Example:**

```php
use Rubix\ML\NeuralNet\CostFunctions\RelativeEntropy;

$costFunction = new RelativeEntropy();
```

---
### Initializers
Initializers are responsible for setting the initial weight parameters of the weight layers of a neural network. Certain activation functions respond differently when given inputs from weight layers with different initializations.

To initialize a random weight matrix:
```php
public initialize(int $fanIn, int $fanOut) : Matrix
```

### Constant
Initialize the parameter to a user specified constant value.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/NeuralNet/Initializers/Constant.php)

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | value | 0. | float | The value to initialize the parameter to. |

**Example:**

```php
use Rubix\ML\NeuralNet\Initializers\Constant;

$initializer = new Constant(1.0);
```

### He
The He initializer was designed for hidden layers that feed into rectified linear unit layers such as [ReLU](#relu), [Leaky ReLU](#leaky-relu), and [ELU](#elu). It draws from a uniform distribution with limits defined as +/- (6 / (fanIn + fanOut)) ** (1. / sqrt(2)).

> [Source](https://github.com/RubixML/RubixML/blob/master/src/NeuralNet/Initializers/He.php)

**Parameters:**

This initializer does not have any parameters.

**Example:**

```php
use Rubix\ML\NeuralNet\Initializers\He;

$initializer = new He();
```

**References:**

>- K. He et al. (2015). Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification.

### Le Cun
Proposed by Yan Le Cun in a paper in 1998, this initializer was one of the first published attempts to control the variance of activations between layers through weight initialization. It remains a good default choice for many hidden layer configurations.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/NeuralNet/Initializers/LeCun.php)

**Parameters:**

This initializer does not have any parameters.

**Example:**

```php
use Rubix\ML\NeuralNet\Initializers\LeCun;

$initializer = new LeCun();
```

**References:**

>- Y. Le Cun et al. (1998). Efficient Backprop.

### Normal
Generates a random weight matrix from a Gaussian distribution with user-specified standard deviation.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/NeuralNet/Initializers/Normal.php)

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | stddev | 0.05 | float | The standard deviation of the distribution to sample from. |

**Example:**

```php
use Rubix\ML\NeuralNet\Initializers\Normal;

$initializer = new Normal(0.1);
```

### Uniform
Generates a random uniform distribution centered at 0 and bounded at both ends by the parameter beta.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/NeuralNet/Initializers/Uniform.php)

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | beta | 0.05 | float | The minimum and maximum bound on the random distribution. |

**Example:**

```php
use Rubix\ML\NeuralNet\Initializers\Uniform;

$initializer = new Uniform(1e-3);
```

### Xavier 1
The Xavier 1 initializer draws from a uniform distribution [-limit, limit] where *limit* is equal to sqrt(6 / (fanIn + fanOut)). This initializer is best suited for layers that feed into an activation layer that outputs a value between 0 and 1 such as [Softmax](#softmax) or [Sigmoid](#sigmoid).

> [Source](https://github.com/RubixML/RubixML/blob/master/src/NeuralNet/Initializers/Xavier1.php)

**Parameters:**

This initializer does not have any parameters.

**Example:**

```php
use Rubix\ML\NeuralNet\Initializers\Xavier1;

$initializer = new Xavier1();
```

**References:**

>- X. Glorot et al. (2010). Understanding the Difficulty of Training Deep Feedforward Neural Networks.

### Xavier 2
The Xavier 2 initializer draws from a uniform distribution [-limit, limit] where *limit* is squal to (6 / (fanIn + fanOut)) ** 0.25. This initializer is best suited for layers that feed into an activation layer that outputs values between -1 and 1 such as [Hyperbolic Tangent](#hyperbolic-tangent) and [Softsign](#softsign).

> [Source](https://github.com/RubixML/RubixML/blob/master/src/NeuralNet/Initializers/Xavier2.php)

**Parameters:**

This initializer does not have any parameters.

**Example:**

```php
use Rubix\ML\NeuralNet\Initializers\Xavier2;

$initializer = new Xavier2();
```

**References:**

>- X. Glorot et al. (2010). Understanding the Difficulty of Training Deep Feedforward Neural Networks.

---
### Layers
Every neural network is made up of layers of computational units called neurons. Each layer processes and transforms the input from the previous layer in such a way that makes it easier for the next layer to form high-level abstractions.

There are three types of layers that form a network, **Input**, **Hidden**, and **Output**. A network can have as many Hidden layers as the user specifies, however, there can only be 1 Input and 1 Output layer per network.

### Input Layers
The entry point for data into a neural network is the input layer which is the first layer in the network. Input layers do not have any learnable parameters.

### Placeholder 1D
The Placeholder 1D input layer represents the *future* input values of a mini batch (matrix) of single dimensional tensors (vectors) to the neural network.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/NeuralNet/Layers/Placeholder1D.php)

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | inputs | None | int | The number of inputs to the neural network. |

**Example:**

```php
use Rubix\ML\NeuralNet\Layers\Placeholder1D;

$layer = new Placeholder1D(100);
```

### Hidden Layers
In multilayer networks, hidden layers are responsible for transforming the input space in such a way that can be linearly separable by the final output layer.

### Activation
Activation layers apply a nonlinear activation function to their inputs.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/NeuralNet/Layers/Activation.php)

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | activation fn | None | object | The function computes the activation of the layer. |

**Example:**

```php
use Rubix\ML\NeuralNet\Layers\Activation;
use Rubix\ML\NeuralNet\ActivationFunctions\ReLU;

$layer = new Activation(new ReLU());
```

### Alpha Dropout
Alpha Dropout is a type of dropout layer that maintains the mean and variance of the original inputs in order to ensure the self-normalizing property of [SELU](#selu) networks with dropout. Alpha Dropout fits with SELU networks by randomly setting activations to the negative saturation value of the activation function at a given ratio each pass.

> **Note**: Alpha Dropout is generally only used in the context of SELU networks. Use regular [Dropout](#dropout) for other types of neural nets.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/NeuralNet/Layers/AlphaDropout.php)

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | ratio | 0.1 | float | The ratio of neurons that are dropped during each training pass. |

**Example:**

```php
use Rubix\ML\NeuralNet\Layers\AlphaDropout;

$layer = new AlphaDropout(0.1);
```

**References:**

>- G. Klambauer et al. (2017). Self-Normalizing Neural Networks.

### Batch Norm
Normalize the activations of the previous layer such that the mean activation is close to 0 and the activation standard deviation is close to 1. Batch Norm can be used to reduce the amount of *covariate shift* within the network making it possible to use higher learning rates and converge faster under some circumstances.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/NeuralNet/Layers/BatchNorm.php)

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | decay | 0.9 | float | The decay rate of the previous running averages of the global mean and variance. |
| 2 | beta initializer | Constant | object | The initializer of the beta parameter. |
| 3 | gamma initializer | Constant | object | The initializer of the gamma parameter. |

**Example:**

```php
use Rubix\ML\NeuralNet\Layers\BatchNorm;
use Rubix\ML\NeuralNet\Initializers\Constant;
use Rubix\ML\NeuralNet\Initializers\Normal;

$layer = new BatchNorm(0.7, new Constant(0.), new Normal(1.));
```

**References:**

>- S. Ioffe et al. (2015). Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift.

### Dense
Dense layers are fully connected neuronal layers, meaning each neuron is connected to each other in the previous layer by a weighted *synapse*. The majority of the parameters in a standard feedforward network are usually contained within the Dense hidden layers of the network.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/NeuralNet/Layers/Dense.php)

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | neurons | None | int | The number of neurons in the layer. |
| 2 | weight initializer | He | object | The initializer of the weight parameter. |
| 3 | bias initializer | Constant | object | The initializer of the bias parameter. |

**Example:**

```php
use Rubix\ML\NeuralNet\Layers\Dense;
use Rubix\ML\NeuralNet\Initializers\He;
use Rubix\ML\NeuralNet\Initializers\Constant;

$layer = new Dense(100, new He(), new Constant(0.));
```

### Dropout
Dropout layers temporarily disable neurons during each training pass. Dropout is a regularization and model averaging technique for reducing overfitting in neural networks by preventing complex co-adaptations on training data.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/NeuralNet/Layers/Dropout.php)

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | ratio | 0.5 | float | The ratio of neurons that are dropped during each training pass. |

**Example:**

```php
use Rubix\ML\NeuralNet\Layers\Dropout;

$layer = new Dropout(0.5);
```

**References:**

>- N. Srivastava et al. (2014). Dropout: A Simple Way to Prevent Neural Networks from Overfitting.

### Noise
This layer adds random Gaussian noise to the inputs to the layer with a standard deviation given as a parameter. Noise added to neural network activations acts as a regularizer by indirectly adding a penalty to the weights through the cost function in the output layer.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/NeuralNet/Layers/Noise.php)

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | stddev | 0.1 | float | The standard deviation of the gaussian noise to add to the inputs. |

**Example:**

```php
use Rubix\ML\NeuralNet\Layers\Noise;

$layer = new Noise(2.0);
```

**References:**

>- C. Gulcehre et al. (2016). Noisy Activation Functions.

### PReLU
The PReLU layer uses leaky ReLU activation functions whose leakage coefficients are parameterized and optimized on a per neuron basis during training.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/NeuralNet/Layers/PReLU.php)

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | initializer | Constant | object | The initializer of the leakage parameter. |

**Example:**

```php
use Rubix\ML\NeuralNet\Layers\PReLU;
use Rubix\ML\NeuralNet\Initializers\Normal;

$layer = new PReLU(new Normal(0.5));
```

**References:**

>- K. He et al. (2015). Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification.

### Output Layers
Activations are read directly from the Output layer when making predictions. The type of output layer will determine the type of Estimator the network can bestow (i.e Binary Classifier, Multiclass Classifier, or Regressor).

### Binary
The Binary layer consists of a single [Sigmoid](#sigmoid) neuron capable of distinguishing between two discrete classes. The Binary layer is useful for neural networks that output a binary class prediction such as *yes* or *no*.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/NeuralNet/Layers/Binary.php)

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | classes | None | array | The unique class labels of the binary classification problem. |
| 2 | alpha | 1e-4 | float | The L2 regularization penalty. |
| 3 | cost fn | Cross Entropy | object | The function that penalizes the activities of bad predictions. |
| 4 | weight initializer | Xavier1 | object | The initializer of the weight parameter. |
| 5 | bias initializer | Constant | object | The initializer of the bias parameter. |

**Example:**

```php
use Rubix\ML\NeuralNet\Layers\Binary;
use Rubix\ML\NeuralNet\CostFunctions\CrossEntropy;

$layer = new Binary(['yes', 'no'], 1e-3, new CrossEntropy());
```

### Continuous
The Continuous output layer consists of a single linear neuron that outputs a scalar value useful for regression problems.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/NeuralNet/Layers/Continuous.php)

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | alpha | 1e-4 | float | The L2 regularization penalty. |
| 2 | cost fn | Least Squares | object | The function that penalizes the activities of bad predictions. |
| 3 | weight initializer | Xavier2 | object | The initializer of the weight parameter. |
| 4 | bias initializer | Constant | object | The initializer of the bias parameter. |

**Example:**

```php
use Rubix\ML\NeuralNet\Layers\Continuous;
use Rubix\ML\NeuralNet\CostFunctions\HuberLoss;

$layer = new Continuous(1e-5, new HuberLoss(3.0));
```

### Multiclass
The Multiclass output layer gives a joint probability estimate of a multiclass classification problem using the [Softmax](#softmax) activation function.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/NeuralNet/Layers/Multiclass.php)

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | classes | None | array | The unique class labels of the multiclass classification problem. |
| 2 | alpha | 1e-4 | float | The L2 regularization penalty. |
| 3 | cost fn | Cross Entropy | object | The function that penalizes the activities of bad predictions. |
| 4 | weight initializer | Xavier1 | object | The initializer of the weight parameter. |
| 5 | bias initializer | Constant | object | The initializer of the bias parameter. |

**Example:**

```php
use Rubix\ML\NeuralNet\Layers\Multiclass;
use Rubix\ML\NeuralNet\CostFunctions\RelativeEntropy;

$layer = new Multiclass(['yes', 'no', 'maybe'], 1e-4, new RelativeEntropy());
```

---
### Optimizers
Gradient Descent is an algorithm that takes iterative steps towards finding the best set of weights in a neural network. Rubix provides a number of pluggable Gradient Descent optimizers that control the step of each parameter in the network.

### AdaGrad
Short for *Adaptive Gradient*, the AdaGrad Optimizer speeds up the learning of parameters that do not change often and slows down the learning of parameters that do enjoy heavy activity.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/NeuralNet/Optimizers/AdaGrad.php)

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | rate | 0.01 | float | The learning rate. i.e. the global step size. |

**Example:**

```php
use Rubix\ML\NeuralNet\Optimizers\AdaGrad;

$optimizer = new AdaGrad(0.125);
```

**References:**

>- J. Duchi et al. (2011). Adaptive Subgradient Methods for Online Learning and Stochastic Optimization.

### Adam
Short for *Adaptive Moment Estimation*, the Adam Optimizer combines both Momentum and RMS prop to achieve a balance of velocity and stability. In addition to storing an exponentially decaying average of past squared gradients like [RMSprop](#rms-prop), Adam also keeps an exponentially decaying average of past gradients, similar to [Momentum](#momentum). Whereas Momentum can be seen as a ball running down a slope, Adam behaves like a heavy ball with friction.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/NeuralNet/Optimizers/Adam.php)

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | rate | 0.001 | float | The learning rate. i.e. the global step size. |
| 2 | momentum decay | 0.1 | float | The decay rate of the accumulated velocity. |
| 3 | norm decay | 0.001 | float | The decay rate of the rms property. |

**Example:**

```php
use Rubix\ML\NeuralNet\Optimizers\Adam;

$optimizer = new Adam(0.0001, 0.1, 0.001);
```

**References:**

>- D. P. Kingma et al. (2014). Adam: A Method for Stochastic Optimization.

### AdaMax
A version of [Adam](#adam) that replaces the RMS property with the infinity norm of the gradients. 

> [Source](https://github.com/RubixML/RubixML/blob/master/src/NeuralNet/Optimizers/AdaMax.php)

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | rate | 0.001 | float | The learning rate. i.e. the global step size. |
| 2 | momentum decay | 0.1 | float | The decay rate of the accumulated velocity. |
| 3 | norm decay | 0.001 | float | The decay rate of the infinity norm. |

**Example:**

```php
use Rubix\ML\NeuralNet\Optimizers\AdaMax;

$optimizer = new AdaMax(0.0001, 0.1, 0.001);
```

**References:**

>- D. P. Kingma et al. (2014). Adam: A Method for Stochastic Optimization.

### Cyclical
The Cyclical optimizer uses a global learning rate that cycles between the lower and upper bound over a designated period while also decaying the upper bound by a factor of gamma each step. Cyclical learning rates have been shown to help escape local minima and saddle points thus achieving higher accuracy.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/NeuralNet/Optimizers/Cyclical.php)

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | lower | 0.001 | float | The lower bound on the learning rate. |
| 2 | upper | 0.006 | float | The upper bound on the learning rate. |
| 3 | steps | 100 | int | The number of steps in every half cycle. |
| 4 | decay | 0.99994 | float | The exponential decay factor to decrease the learning rate by every step. |

**Example:**

```php
use Rubix\ML\NeuralNet\Optimizers\Cyclical;

$optimizer = new Cyclical(0.001, 0.005, 1000);
```

**References:**

>- L. N. Smith. (2017). Cyclical Learning Rates for Training Neural Networks.

### Momentum
Momentum adds velocity to each step until exhausted. It does so by accumulating momentum from past updates and adding a factor of the previous velocity to the current step.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/NeuralNet/Optimizers/Momentum.php)

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | rate | 0.001 | float | The learning rate. i.e. the global step size. |
| 2 | decay | 0.1 | float | The decay rate of the accumulated velocity. |

**Example:**

```php
use Rubix\ML\NeuralNet\Optimizers\Momentum;

$optimizer = new Momentum(0.001, 0.2);
```

**References:**

>- D. E. Rumelhart et al. (1988). Learning representations by back-propagating errors.

### RMS Prop
An adaptive gradient technique that divides the current gradient over a rolling window of magnitudes of recent gradients.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/NeuralNet/Optimizers/RMSProp.php)

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | rate | 0.001 | float | The learning rate. i.e. the global step size. |
| 2 | decay | 0.1 | float | The decay rate of the rms property. |

**Example:**

```php
use Rubix\ML\NeuralNet\Optimizers\RMSProp;

$optimizer = new RMSProp(0.01, 0.1);
```

**References:**

>- T. Tieleman et al. (2012). Lecture 6e rmsprop: Divide the gradient by a running average of its recent magnitude.

### Step Decay
A learning rate decay optimizer that reduces the learning rate by a factor of the decay parameter whenever it reaches a new *floor*. The number of steps needed to reach a new floor is defined by the *steps* parameter.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/NeuralNet/Optimizers/StepDecay.php)

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | rate | 0.01 | float | The learning rate. i.e. the global step size. |
| 2 | steps | 100 | int | The size of every floor in steps. i.e. the number of steps to take before applying another factor of decay. |
| 3 | decay | 1e-3 | float | The factor to decrease the learning rate at each *floor*. |

**Example:**

```php
use Rubix\ML\NeuralNet\Optimizers\StepDecay;

$optimizer = new StepDecay(0.1, 50, 1e-3);
```

### Stochastic
A constant learning rate optimizer based on the original Stochastic Gradient Descent paper.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/NeuralNet/Optimizers/Stochastic.php)

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | rate | 0.01 | float | The learning rate. i.e. the global step size. |

**Example:**

```php
use Rubix\ML\NeuralNet\Optimizers\Stochastic;

$optimizer = new Stochastic(0.01);
```

---
### Kernels

### Distance
Distance kernels measure the distance between points in vector space. They are used throughout Rubix in Estimators that employ the concept of *distance* to make predictions such as [K Nearest Neighbors](#k-nearest-neighbors), [K Means](#k-means), and [Local Outlier Factor](#local-outlier-factor).

> **Note**: Distance is only defined for *continuous* data points in Rubix.

### Canberra
A weighted version of [Manhattan](#manhattan) distance which computes the L1 distance between two coordinates in a vector space.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Kernels/Distance/Canberra.php)

**Parameters:**

This kernel does not have any parameters.

**Example:**

```php
use Rubix\ML\Kernels\Distance\Canberra;

$kernel = new Canberra();
```

### Cosine
Cosine Similarity is a measure that ignores the magnitude of the distance between two vectors thus acting as strictly a judgement of orientation. Two vectors with the same orientation have a cosine similarity of 1, two vectors oriented at 90° relative to each other have a similarity of 0, and two vectors diametrically opposed have a similarity of -1. To be used as a distance function, we subtract the Cosine Similarity from 1 in order to satisfy the positive semi-definite condition, therefore the Cosine *distance* is a number between 0 and 2.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Kernels/Distance/Cosine.php)

**Parameters:**

This kernel does not have any parameters.

**Example:**

```php
use Rubix\ML\Kernels\Distance\Cosine;

$kernel = new Cosine();
```

### Diagonal
The Diagonal (sometimes called *Chebyshev*) distance is a measure that constrains movement to horizontal, vertical, and diagonal from a point. An example that uses Diagonal movement is a chess board.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Kernels/Distance/Diagonal.php)

**Parameters:**

This kernel does not have any parameters.

**Example:**

```php
use Rubix\ML\Kernels\Distance\Diagonal;

$kernel = new Diagonal();
```

### Euclidean
This is the ordinary straight line (*bee line*) distance between two points in Euclidean space. The associated norm of the Euclidean distance is called the L2 norm.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Kernels/Distance/Euclidean.php)

**Parameters:**

This kernel does not have any parameters.

**Example:**

```php
use Rubix\ML\Kernels\Distance\Euclidean;

$kernel = new Euclidean();
```

### Jaccard
This *generalized* Jaccard distance is a measure of similarity that one sample has to another with a range from 0 to 1. The higher the percentage, the more dissimilar they are.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Kernels/Distance/Jaccard.php)

**Parameters:**

This kernel does not have any parameters.

**Example:**

```php
use Rubix\ML\Kernels\Distance\Jaccard;

$kernel = new Jaccard();
```

### Manhattan
A distance metric that constrains movement to horizontal and vertical, similar to navigating the city blocks of Manhattan. An example that used this type of movement is a checkers board.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Kernels/Distance/Manhattan.php)

**Parameters:**

This kernel does not have any parameters.

**Example:**

```php
use Rubix\ML\Kernels\Distance\Manhattan;

$kernel = new Manhattan();
```

### Minkowski
The Minkowski distance is a metric in a normed vector space which can be considered as a generalization of both the [Euclidean](#euclidean) and [Manhattan](#manhattan) distances. When the *lambda* parameter is set to 1 or 2, the distance is equivalent to Manhattan and Euclidean respectively.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Kernels/Distance/Minkowski.php)

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | lambda | 3.0 | float | Controls the curvature of the unit circle drawn from a point at a fixed distance. |

**Example:**

```php
use Rubix\ML\Kernels\Distance\Minkowski;

$kernel = new Minkowski(4.0);
```

---
### SVM
Support Vector Machine kernels are used in the context of SVM-based estimators to project sample vectors into a non-linear feature space, allowing them to marginalize non-linear data.

### Linear
A simple linear kernel computed by the dot product of two vectors.

**Parameters:**

This kernel does not have any parameters.

**Example:**

```php
use Rubix\ML\Kernels\SVM\Linear;

$kernel = new Linear();
```

### Polynomial
This kernel projects a sample vector using polynomials of the p'th degree.

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | degree | 3 | int | The degree of the polynomial. |
| 2 | gamma | null | float | The kernel coefficient. |
| 3 | coef0 | 0. | float | The independent term. |

**Example:**

```php
use Rubix\ML\Kernels\SVM\Polynomial;

$kernel = new Polynomial(3, null, 0.);
```

### RBF
Non linear radial basis function computes the distance from a centroid or origin.

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | gamma | null | float | The kernel coefficient. |

**Example:**

```php
use Rubix\ML\Kernels\SVM\RBF;

$kernel = new RBF(null);
```

### Sigmoidal
S shaped nonliearity kernel with output values ranging from -1 to 1.

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | gamma | null | float | The kernel coefficient. |
| 2 | coef0 | 0. | float | The independent term. |

**Example:**

```php
use Rubix\ML\Kernels\SVM\Sigmoidal;

$kernel = new Sigmoidal(null, 0.);
```

---
### Cross Validation
Cross validation is the process of testing the generalization performance of a model.

### Validators
Validators take an [Estimator](#estimators) instance, [Labeled Dataset](#labeled) object, and validation [Metric](#validation-metrics) and return a validation score that measures the generalization performance of the model using one of various cross validation techniques. There is no need to train the Estimator beforehand as the Validator will automatically train it on subsets of the dataset created by the testing algorithm.

```php
public test(Estimator $estimator, Labeled $dataset, Validation $metric) : float
```

**Example:**

```php
use Rubix\ML\CrossValidation\KFold;
use Rubix\ML\CrossValidation\Metrics\Accuracy;

$validator = new KFold(10);

$score = $validator->test($estimator, $dataset, new Accuracy());

var_dump($score);
```

**Output:**
```sh
float(0.869)
```

### Hold Out
Hold Out is a simple cross validation technique that uses a *hold out* validation set. The advantages of Hold Out is that it is quick, but it doesn't allow the model to train on the entire training set.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/CrossValidation/HoldOut.php)

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | ratio | 0.2 | float | The ratio of samples to hold out for testing. |
| 2 | stratify | false | bool | Should we stratify the dataset before splitting? |

**Example:**

```php
use Rubix\ML\CrossValidation\HoldOut;

$validator = new HoldOut(0.25, true);
```

### K Fold
K Fold is a technique that splits the training set into K individual sets and for each training round uses 1 of the folds to measure the validation performance of the model. The score is then averaged over K. For example, a K value of 10 will train and test 10 versions of the model using a different testing set each time.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/CrossValidation/KFold.php)

**Interfaces:** [Parallel](#parallel)

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | k | 10 | int | The number of times to split the training set into equal sized folds. |
| 2 | stratify | false | bool | Should we stratify the dataset before folding? |

**Example:**

```php
use Rubix\ML\CrossValidation\KFold;

$validator = new KFold(5, true);
```

### Leave P Out
Leave P Out tests the model with a unique holdout set of P samples for each round until all samples have been tested.

> **Note**: Leave P Out can become slow with large datasets and small values of P.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/CrossValidation/LeavePOut.php)

**Interfaces:** [Parallel](#parallel)

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | p | 10 | int | The number of samples to leave out each round for testing. |

**Example:**

```php
use Rubix\ML\CrossValidation\LeavePOut;

$validator = new LeavePOut(50);
```

### Monte Carlo
Repeated Random Subsampling or Monte Carlo cross validation is a technique that takes the average validation score over a user-supplied number of simulations (randomized splits of the dataset).

> [Source](https://github.com/RubixML/RubixML/blob/master/src/CrossValidation/MonteCarlo.php)

**Interfaces:** [Parallel](#parallel)

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | simulations | 10 | int | The number of simulations to run i.e the number of tests to average. |
| 2 | ratio | 0.2 | float | The ratio of samples to hold out for testing. |
| 3 | stratify | false | bool | Should we stratify the dataset before splitting? |

**Example:**

```php
use Rubix\ML\CrossValidation\MonteCarlo;

$validator = new MonteCarlo(30, 0.1);
```

### Validation Metrics
Validation metrics are for evaluating the performance of an Estimator given the ground truth labels.

Compute a validation score, pass in the predictions from an estimator along with the ground-truth labels:
```php
public score(array $predictions, array $labels) : float
```

Output the range of values the validation score can take on in a 2-tuple:
```php
public range() : array
```

Return a list of estimators that metric is compatible with:
```php
public compatibility() : array
```

**Example:**

```php
use Rubix\ML\CrossValidation\Metrics\MeanAbsoluteError;

$metric = new MeanAbsoluteError();

$score = $metric->score($predictions, $labels);

var_dump($metric->range());

var_dump($score);
```

**Output:**
```sh
array(2) {
  [0]=> float(-INF)
  [1]=> int(0)
}

float(-0.99846070553066)
```

### Accuracy
Accuracy is a quick classification and anomaly detection metric defined as the number of true positives over all samples in the testing set.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/CrossValidation/Metrics/Accuracy.php)

**Compatibility:** Classification, Anomaly Detection

**Range:** 0 to 1

**Example:**

```php
use Rubix\ML\CrossValidation\Metrics\Accuracy;

$metric = new Accuracy();
```

### Completeness
A ground truth clustering metric that measures the ratio of samples in a class that are also members of the same cluster. A cluster is said to be *complete* when all the samples in a class are contained in a cluster.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/CrossValidation/Metrics/Completeness.php)

**Compatibility:** Clustering

**Range:** 0 to 1

**Example:**

```php
use Rubix\ML\CrossValidation\Metrics\Completeness;

$metric = new Completeness();
```

### F Beta
A weighted harmonic mean of precision and recall. The beta parameter controls the weight of precision in the combined score. As beta goes to infinity the score only considers recall whereas when it goes to 0 it only considers precision.

> **Note**: Beta equal to 1 is called an F1 score.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/CrossValidation/Metrics/FBeta.php)

**Compatibility:** Classification, Anomaly Detection

**Range:** 0 to 1

**Parameters**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | beta | 1. | float | The weight of precision in the harmonic mean. |

**Example:**

```php
use Rubix\ML\CrossValidation\Metrics\FBeta;

$metric = new FBeta(0.7);
```

### Homogeneity
A ground truth clustering metric that measures the ratio of samples in a cluster that are also members of the same class. A cluster is said to be *homogeneous* when the entire cluster is comprised of a single class of samples.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/CrossValidation/Metrics/Homogeneity.php)

**Compatibility:** Clustering

**Range:** 0 to 1

**Example:**

```php
use Rubix\ML\CrossValidation\Metrics\Homogeneity;

$metric = new Homogeneity();
```

### MCC
Matthews Correlation Coefficient measures the quality of a classification. It takes into account true and false positives and negatives and is generally regarded as a balanced measure which can be used even if the classes are of very different sizes. The MCC is in essence a correlation coefficient between the observed and predicted binary classifications; it returns a value between −1 and +1. A coefficient of +1 represents a perfect prediction, 0 no better than random prediction and −1 indicates total disagreement between prediction and observation.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/CrossValidation/Metrics/MCC.php)

**Compatibility:** Classification, Anomaly Detection

**Range:** -1 to 1

**Example:**

```php
use Rubix\ML\CrossValidation\Metrics\MCC;

$metric = new MCC();
```

### Mean Absolute Error
A metric that measures the average amount that a prediction is off by given some ground truth (labels).

> [Source](https://github.com/RubixML/RubixML/blob/master/src/CrossValidation/Metrics/MeanAbsoluteError.php)

**Compatibility:** Regression

**Range:** -∞ to 0

**Example:**

```php
use Rubix\ML\CrossValidation\Metrics\MeanAbsoluteError;

$metric = new MeanAbsoluteError();
```

### Mean Squared Error
A regression metric that punishes bad predictions the worse they get by averaging the *squared* error  over the testing set.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/CrossValidation/Metrics/MeanSquaredError.php)

**Compatibility:** Regression

**Range:** -∞ to 0

**Example:**

```php
use Rubix\ML\CrossValidation\Metrics\MeanSquaredError;

$metric = new MeanSquaredError();
```

### Median Absolute Error
Median Absolute Error (MAE) is a robust measure of the error that ignores highly erroneous predictions.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/CrossValidation/Metrics/MedianAbsoluteError.php)

**Compatibility:** Regression

**Range:** -∞ to 0

**Example:**

```php
use Rubix\ML\CrossValidation\Metrics\MedianAbsoluteError;

$metric = new MedianAbsoluteError();
```

### Rand Index
The Adjusted Rand Index is a measure of similarity between the clustering and some ground truth that is adjusted for chance. It considers all pairs of samples that are assigned in the same or different clusters in the predicted and empirical clusterings.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/CrossValidation/Metrics/RandIndex.php)

**Compatibility:** Regression

**Range:** -1 to 1

**Example:**

```php
use Rubix\ML\CrossValidation\Metrics\RandIndex;

$metric = new RandIndex();
```

**References:**

>- W. M. Rand. (1971). Objective Criteria for the Evaluation of  Clustering Methods.

### RMS Error
Root Mean Squared (RMS) Error or average L2 loss is a metric that is used to compute the residuals of a regression problem.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/CrossValidation/Metrics/RMSError.php)

**Compatibility:** Regression

**Range:** -∞ to 0

**Example:**

```php
use Rubix\ML\CrossValidation\Metrics\RMSError;

$metric = new RMSError();
```

### R Squared
The *coefficient of determination* or R Squared (R²) is the proportion of the variance in the dependent variable that is predictable from the independent variable(s).

> [Source](https://github.com/RubixML/RubixML/blob/master/src/CrossValidation/Metrics/RSquared.php)

**Compatibility:** Regression

**Range:** -∞ to 1

**Example:**

```php
use Rubix\ML\CrossValidation\Metrics\RSquared;

$metric = new RSquared();
```

### V Measure
V Measure is the harmonic balance between [homogeneity](#homogeneity) and [completeness](#completeness) and is used as a measure to determine the quality of a clustering.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/CrossValidation/Metrics/VMeasure.php)

**Compatibility:** Clustering

**Range:** 0 to 1

**Example:**

```php
use Rubix\ML\CrossValidation\Metrics\VMeasure;

$metric = new VMeasure();
```

**References:**

>- A. Rosenberg et al. (2007). V-Measure: A conditional entropy-based external cluster evaluation measure.

---
### Reports
Reports offer a comprehensive view of the performance of an estimator given the problem in question.

To generate a report from the predictions of an estimator given some ground truth labels:
```php
public generate(array $predictions, array $labels) : array
```

Return a list of estimators that report is compatible with:
```php
public compatibility() : array
```

**Example:**

```php
use Rubix\ML\Reports\ConfusionMatrix;

$report = new ConfusionMatrix();

$result = $report->generate($predictions, $labels);
```

### Aggregate Report
A report that aggregates the results of multiple reports. The reports are indexed by the key given at construction time.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/CrossValidation/Reports/AggregateReport.php)

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | reports | | array | An array of report objects to aggregate. |

**Example:**

```php
use Rubix\ML\CrossValidation\Reports\AggregateReport;
use Rubix\ML\CrossValidation\Reports\ConfusionMatrix;
use Rubix\ML\CrossValidation\Reports\MulticlassBreakdown;

$report = new AggregateReport([
	'breakdown' => new MulticlassBreakdown(),
	'matrix' => new ConfusionMatrix(),
]);

$result = $report->generate($estimator, $testing);
```

### Confusion Matrix
A Confusion Matrix is a table that visualizes the true positives, false, positives, true negatives, and false negatives of a classifier. The name stems from the fact that the matrix makes it easy to see the classes that the classifier might be confusing.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/CrossValidation/Reports/ConfusionMatrix.php)

**Compatibility:** Classification, Anomaly Detection

**Parameters:**

This report does not have any parameters.

**Example:**

```php
use Rubix\ML\CrossValidation\Reports\ConfusionMatrix;

$report = new ConfusionMatrix();

$result = $report->generate($estimator, $testing);

var_dump($result);
```

**Output:**

```sh
  array(3) {
    ["dog"]=> array(2) {
      ["dog"]=> int(842)
      ["cat"]=> int(5)
      ["turtle"]=> int(0)
    }
    ["cat"]=>
    array(2) {
      ["dog"]=> int(0)
      ["cat"]=> int(783)
      ["turtle"]=> int(3)
    }
    ["turtle"]=>
    array(2) {
      ["dog"]=> int(31)
      ["cat"]=> int(79)
      ["turtle"]=> int(496)
    }
  }
```

### Contingency Table
A Contingency Table is used to display the frequency distribution of class labels among a clustering of samples. It is similar to a [Confusion Matrix](#confusion-matrix) but uses the labels to establish a ground truth for a clustering instead.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/CrossValidation/Reports/ContingencyTable.php)

**Compatibility:** Clustering

**Parameters:**

This report does not have any parameters.

**Example:**

```php
use Rubix\ML\CrossValidation\Reports\ContingencyTable;

$report = new ContingencyTable();

$result = $report->generate($estimator, $testing);

var_dump($result);
```

**Output:**

```sh
array(3) {
    [1]=>
    array(3) {
      [1]=> int(13)
      [2]=> int(0)
      [3]=> int(2)
    }
    [2]=>
    array(3) {
      [1]=> int(1)
      [2]=> int(0)
      [3]=> int(12)
    }
    [0]=>
    array(3) {
      [1]=> int(0)
      [2]=> int(14)
      [3]=> int(0)
    }
  }
```

### Multiclass Breakdown
A report that drills down in to each unique class outcome. The report includes metrics such as Accuracy, F1 Score, MCC, Precision, Recall, Fall Out, and Miss Rate.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/CrossValidation/Reports/MulticlassBreakdown.php)

**Compatibility:** Classification, Anomaly Detection

**Parameters:**

This report does not have any parameters.

**Example:**

```php
use Rubix\ML\CrossValidation\Reports\MulticlassBreakdown;

$report = new MulticlassBreakdown();

$result = $report->generate($estimator, $testing);

var_dump($result);
```

**Output:**

```sh
["label"]=> array(2) {
	["wolf"]=> array(19) {
      	["accuracy"]=> float(0.6)
      	["precision"]=> float(0.66666666666667)
      	["recall"]=> float(0.66666666666667)
      	["specificity"]=> float(0.5)
      	["negative_predictive_value"]=> float(0.5)
      	["false_discovery_rate"]=> float(0.33333333333333)
      	["miss_rate"]=> float(0.33333333333333)
      	["fall_out"]=> float(0.5)
      	["false_omission_rate"]=> float(0.5)
     	["f1_score"]=> float(0.66666666666667)
      	["mcc"]=> float(0.16666666666667)
      	["informedness"]=> float(0.16666666666667)
      	["markedness"]=> float(0.16666666666667)
      	["true_positives"]=> int(2)
      	["true_negatives"]=> int(1)
      	["false_positives"]=> int(1)
      	["false_negatives"]=> int(1)
      	["cardinality"]=> int(3)
      	["density"]=> float(0.6)
    }
```

### Residual Analysis
Residual Analysis is a Report that measures the differences between the predicted and actual values of a regression problem in detail.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/CrossValidation/Reports/ResidualAnalysis.php)

**Compatibility:** Regression

**Parameters:**

This report does not have any parameters.

**Example:**

```php
use Rubix\ML\CrossValidation\Reports\ResidualAnaysis;

$report = new ResidualAnalysis();

$result = $report->generate($estimator, $testing);

var_dump($result);
```

**Output:**

```sh
  array(12) {
    ["mean_absolute_error"]=> float(0.44927554249285)
    ["median_absolute_error"]=> float(0.30273889978541)
    ["mean_squared_error"]=> float(0.44278193357447)
    ["rms_error"]=> float(0.66541861529001)
	["mean_squared_log_error"]=> float(-0.35381010755)
	["r_squared"]=> float(0.99393263320234)
    ["error_mean"]=> float(0.14748941084881)
    ["error_variance"]=> float(0.42102880726195)
    ["error_skewness"]=> float(-2.7901397847317)
    ["error_kurtosis"]=> float(12.967400285518)
    ["error_min"]=> float(-3.5540079974946)
    ["error_max"]=> float(1.4097829828182)
    ["cardinality"]=> int(80)
  }
```

---
### Other
This section includes all classes that do not fall under a specific category.

### Strategies
Guesses can be thought of as a type of *weak* prediction. Unlike a real prediction, guesses are made using limited information. A guessing Strategy attempts to use such information to formulate an educated guess. Guessing is utilized in both Dummy Estimators ([Dummy Classifier](#dummy-classifier), [Dummy Regressor](#dummy-regressor)) as well as the [Missing Data Imputer](#missing-data-imputer).

The Strategy interface provides an API similar to Transformers as far as fitting, however, instead of being fit to an entire dataset, each Strategy is fit to an array of either continuous or discrete values.

To fit a Strategy to an array of values:
```php
public fit(array $values) : void
```

To make a guess based on the fitted data:
```php
public guess() : mixed
```

### Blurry Percentile
A strategy that guesses within the domain of the p-th percentile of the fitted data plus some gaussian noise.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Other/Strategies/BlurryPercentile.php)

**Data Type:** Continuous

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | p | 50.0 | float | The index of the percentile to predict where 50 is the median. |
| 2 | blur | 0.1 | float | The amount of gaussian noise to add to the guess as a factor of the median absolute deviation (MAD). |

**Example:**

```php
use Rubix\ML\Other\Strategies\BlurryPercentile;

$strategy = new BlurryPercentile(34.0, 0.2);
```

### Constant
Always guess a constant value.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Other/Strategies/Constant.php)

**Data Type:** Continuous

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | value | 0. | float | The value to guess. |

**Example:**
```php
use Rubix\ML\Other\Strategies\Constant;

$strategy = new Constant(17.);
```

### K Most Frequent
This strategy outputs one of K most frequent discrete values at random.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Other/Strategies/KMostFrequent.php)

**Data Type:** Categorical

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | k | 1 | int | The number of most frequency categories to consider when formulating a guess. |

**Example:**

```php
use Rubix\ML\Other\Strategies\KMostFrequent;

$strategy = new KMostFrequent(5);
```

### Lottery
Hold a lottery in which each category has an equal chance of being picked.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Other/Strategies/Lottery.php)

**Data Type:** Categorical

**Parameters:**

This strategy does not have any parameters.

**Example:**

```php
use Rubix\ML\Other\Strategies\Lottery;

$strategy = new Lottery();
```

### Mean
This strategy always predicts the mean of the fitted data.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Other/Strategies/Mean.php)

**Data Type:** Continuous

**Parameters:**

This strategy does not have any parameters.

**Example:**

```php
use Rubix\ML\Other\Strategies\Mean;

$strategy = new Mean();
```

### Popularity Contest
Hold a popularity contest where the probability of winning (being guessed) is based on the category's prior probability.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Other/Strategies/PopularityContest.php)

**Data Type:** Categorical

**Parameters:**

This strategy does not have any parameters.

**Example:**

```php
use Rubix\ML\Other\Strategies\Lottery;

$strategy = new PopularityContest();
```

### Wild Guess
It is what you think it is. Make a guess somewhere in between the minimum and maximum values observed during fitting with equal probability given to all values within range.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Other/Strategies/WildGuess.php)

**Data Type:** Continuous

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | precision | 2 | int | The number of decimal places of precision for each guess. |

**Example:**

```php
use Rubix\ML\Other\Strategies\WildGuess;

$strategy = new WildGuess(5);
```

### Helpers

### Data Type
Determine the data type of a variable according to Rubix ML's type system.

To determine the data type of a variable:
```php
public determine($variable) : int
```

> **Note**: The return value is an integer encoding of the datatype defined as constants on the DataType class.

Return true if the variable is categorical:
```php
public isCategorical($variable) : bool
```

Return true if the variable is categorical:
```php
public isContinuous($variable) : bool
```

Return true if the variable is a PHP resource:
```php
public isResource($variable) : bool
```

Return true if the variable is an unrecognized data type:
```php
public isOther($variable) : bool
```

**Example:**

```php
use Rubix\ML\Other\Helpers\DataType;

var_dump(DataType::determine('string'));

var_dump(DataType::isContinuous(16));

var_dump(DataType::isCategorical(18));
```

**Output:**

```sh
int(2) // Categorical
bool(true)
bool(false)
```

### Params
Generate distributions of values to use in conjunction with [Grid Search](#grid-search) or other forms of model selection and/or cross validation.

To generate a *unique* distribution of integer parameters:
```php
public static ints(int $min, int $max, int $n = 10) : array
```

To generate a random distribution of floating point parameters:
```php
public static floats(float $min, float $max, int $n = 10) : array
```

To generate a uniformly spaced grid of parameters:
```php
public static grid(float $min, float $max, int $n = 10) : array
```

**Example:**

```php
use Rubix\ML\Other\Helpers\Params;

$ints = Params::ints(0, 100, 5);

$floats = Params::floats(0, 100, 5);

$grid = Params::grid(0, 100, 5);

var_dump($ints);
var_dump($floats);
var_dump($grid);
```

**Output:**

```sh
array(5) {
  [0]=> int(88)
  [1]=> int(48)
  [2]=> int(64)
  [3]=> int(100)
  [4]=> int(41)
}

array(5) {
  [0]=> float(42.65728411)
  [1]=> float(66.74335233)
  [2]=> float(15.1724384)
  [3]=> float(71.92631156)
  [4]=> float(4.63886342)
}

array(5) {
  [0]=> float(0)
  [1]=> float(25)
  [2]=> float(50)
  [3]=> float(75)
  [4]=> float(100)
}

```

**Example:**

```php
use Rubix\ML\GridSearch;
use Rubix\ML\Other\Helpers\Params;
use Rubix\ML\Clusterers\FuzzyCMeans;
use Rubix\ML\Kernels\Distance\Diagonal;
use Rubix\ML\Kernels\Distance\Minkowski;
use Rubix\CrossValidation\KFold;
use Rubix\CrossValidation\Metrics\VMeasure;

$params = [
	Params::grid(1, 5, 5), Params::floats(1.0, 20.0, 20), [new Diagonal(), new Minkowski(3.0)],
];

$estimator = new GridSearch(FuzzyCMeans::class, $params, new VMeasure(), new KFold(10));

$estimator->train($dataset);

var_dump($estimator->best());
```

**Output:**

```sh
array(3) {
  [0]=> int(4)
  [1]=> float(13.65)
  [2]=> object(Rubix\ML\Kernels\Distance\Diagonal)#15 (0) {
  }
}
```

### Loggers

> **Note**: All loggers implement the standard PSR-3 interface.

### Screen
A logger that outputs to the php standard output.

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | channel | 'default' | string | The channel name that appears on each line. |
| 2 | timestamps | true | bool | Should we show timestamps? |

**Example:**

```php
use Rubix\ML\Other\Loggers\Screen;

$logger = new Screen('credit', true);
```

---
### Tokenizers
Tokenizers take a body of text and convert the words to an array of string *tokens*. Tokens can represent a single word or multiple words such as in [NGram](#n-gram) and [SkipGram](#skip-gram). Tokenizers are used by various transformers in Rubix such as the [Word Count Vectorizer](#word-count-vectorizer) to represent blobs of text as token counts.

To tokenize a blob of text:
```php
public tokenize(string $text) : array
```

**Example:**
```php
use Rubix\ML\Extractors\Tokenizers\Word;

$text = 'I would like to die on Mars, just not on impact.';

$tokenizer = new Word();

var_dump($tokenizer->tokenize($text));
```

**Output:**

```sh
  array(10) {
    [0]=> string(5) "would"
		[1]=> string(4) "like"
		[2]=> string(2) "to"
		[3]=> string(3) "die"
		[4]=> string(2) "on"
		[5]=> string(4) "Mars"
		[6]=> string(4) "just"
		[7]=> string(3) "not"
		[8]=> string(2) "on"
		[9]=> string(6) "impact"
  }
```

### N-gram
N-grams are sequences of n-words of a given string. The N-gram tokenizer outputs tokens of contiguous words ranging from *min* to *max* number of words per token.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Other/Tokenizers/NGram.php)

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | min | 2 | int | The minimum number of contiguous words to a token. |
| 2 | max | 2 | int | The maximum number of contiguous words to a token. |

**Example:**

```php
use Rubix\ML\Extractors\Tokenizers\NGram;

$tokenizer = new NGram(1, 3);
```

### Skip-Gram
Skip-grams are a technique similar to n-grams, whereby n-grams are formed but in addition to allowing adjacent sequences of words, the next *k* words will be *skipped* forming n-grams of the new forward looking sequences.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Other/Tokenizers/SkipGram.php)

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | n | 2 | int | The number of contiguous words to a single token. |
| 2 | skip | 2 | int | The number of words to skip over to form new n-gram sequences. |

**Example:**

```php
use Rubix\ML\Extractors\Tokenizers\SkipGram;

$tokenizer = new SkipGram(2, 2);
```

### Whitespace
Tokens are delimited by a user-specified whitespace character.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Other/Tokenizers/Whitespace.php)

**Parameters:**

| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | delimiter | ' ' | string | The whitespace character that delimits each token. |

**Example:**

```php
use Rubix\ML\Extractors\Tokenizers\Whitespace;

$tokenizer = new Whitespace(',');
```

### Word Tokenizer
The Word tokenizer uses a regular expression to tokenize the words in a blob of text.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Other/Tokenizers/Word.php)

**Parameters:**

This tokenizer does not have any parameters.

**Example:**

```php
use Rubix\ML\Extractors\Tokenizers\Word;

$tokenizer = new Word();
```

---
## FAQ
Here you can find answers to the most frequently asked Rubix ML questions.

### What environment (SAPI) should I run Rubix in?
All Rubix programs are designed to run from the PHP [command line interface](http://php.net/manual/en/features.commandline.php) (CLI). The reason almost always boils down to performance and memory consumption.

If you want to serve your trained estimators in production then you can use the [Rubix Server](https://github.com/RubixML/Server) library to run a standalone model server that implements its own networking (HTTP, TCP, ZMQ, etc.) layer and runs from the CLI instead of Apache server or NGINX via FPM which is much slower.

To run a program using the PHP command line interface (CLI), open a terminal and enter:
```sh
$ php example.php
```

> **Note**: The PHP interpreter must be in your default PATH for the above syntax to work.

### I'm getting out of memory errors
Try adjusting the `memory_limit` option in your php.ini file to something more reasonable. We recommend setting this to *-1* (no limit) or slightly below your device's memory supply for best results.

> **Note**: Machine Learning can sometimes require a lot of memory. The amount necessary will depend on the amount of training data and the size of your model. If you have more data than you can hold in memory, some learners allow you to train in batches. See the section on [Online](#online) estimators for more information.

### What is a Tuple?
A *tuple* is a way to denote an immutable sequential array with a predefined length. An *n-tuple* is a tuple with the length of n. In other languages, tuples are a separate datatype and their properties such as immutability are enforced by the compiler/interpreter, unlike PHP arrays.

**Example:**

```php
$tuple = ['first', 'second', 0.001]; // a 3-tuple
```

### What is the difference between categorical and continuous data types?
There are generally 2 classes of data types that Rubix distinguishes by convention.

Categorical (or *discrete*) data are those that describe a *qualitative* property of a sample such as *gender* or *city* and can be 1 of K possible values. Categorical feature types are always represented as a string internal type.

Continuous data are *quantitative* properties of samples such as *age* or *speed* and can be any number within the set of infinite real numbers. Continuous features are represented as either floating point or integer types internally.

### Does Rubix support multiprocessing?
Yes, Rubix currently supports parallel processing (multiprocessing) by utilizing various parllel computing [Backends](#backends) under the hood of objects that implement the [Parallel](#parallel) interface.

### Does Rubix support multithreading?
Not currently, however we plan to add CPU and GPU multithreading in the future.

### Does Rubix support Deep Learning?
Yes. Deep Learning is a subset of machine learning that involves forming higher-order representations of the input features such as edges and textures in an image or word meanings. A number of learners in Rubix support Deep Learning including the [Multi Layer Perceptron](#multi-layer-perceptron) classifier and [MLP Regressor](#mlp-regressor).

### Does Rubix support Reinforcement Learning?
Not currently. Rubix is for *supervised* and *unsupervised* learning only.

---
## Testing
Rubix utilizes a combination of static code analysis and unit tests for quality assurance (QA) and to reduce the number of bugs. Rubix provides three [Composer](https://getcomposer.org/) scripts that can be run from the root directory to automate the testing process.

To run static analysis:
```sh
$ composer analyze
```

To run the style checker:
```sh
$ composer check
```

To run the unit tests:
```sh
$ composer test
```

---
## Contributing
See [CONTRIBUTING.md](https://github.com/RubixML/RubixML/blob/master/CONTRIBUTING.md) for guidelines.

---
## License
[MIT](https://github.com/RubixML/RubixML/blob/master/LICENSE.md)
