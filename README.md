Welcome file
Welcome file
# Rubix ML for PHP
[![PHP from Packagist](https://img.shields.io/packagist/php-v/rubix/ml.svg?style=for-the-badge)](https://www.php.net/) [![Latest Stable Version](https://img.shields.io/packagist/v/rubix/ml.svg?style=for-the-badge)](https://packagist.org/packages/rubix/ml) [![Travis](https://img.shields.io/travis/andrewdalpino/Rubix-ML.svg?style=for-the-badge)](https://travis-ci.org/andrewdalpino/Rubix-ML) [![GitHub license](https://img.shields.io/github/license/andrewdalpino/Rubix.svg?style=for-the-badge)](https://github.com/andrewdalpino/Rubix/blob/master/LICENSE.md)

Rubix ML is a library that lets you build intelligent programs that learn from data in [PHP](https://php.net).

## Our Mission
The goal of Rubix is to bring easy to use machine learning (ML) capabilities to the PHP language. We aspire to provide the framework to facilitate small to medium sized projects, rapid prototyping, and education. If you would like to join in on the mission, you get up and running fast by following the instructions below.

## Installation
Install Rubix using composer:
```sh
$ composer require rubix/ml
```

## Requirements
- [PHP](https://php.net) 7.1.3 or above
- [GD extension](https://php.net/manual/en/book.image.php) for Image Vectorization

## License
MIT

## Documentation

### Table of Contents

 - [Introduction](#an-introduction-to-machine-learning-in-rubix)
	 - [Obtaining Data](#obtaining-data)
	 - [Choosing an Estimator](#choosing-an-estimator)
	 - [Training and Prediction](#training-and-prediction)
	 - [Evaluation](#evaluating-model-performance)
	 - [Next Steps](#what-next)
- [API Reference](#api-reference)
	- [Datasets](#datasets)
		- [Dataset Objects](#dataset-objects)
			- [Labeled](#labeled)
			- [Unlabeled](#unlabeled)
		- [Generators](#generators)
			- [Agglomerate](#agglomerate)
			- [Blob](#blob)
			- [Circle](#circle)
			- [Half Moon](#half-moon)
	- [Feature Extractors](#feature-extractors)
    	- [Count Vectorizer](#count-vectorizer)
		- [Pixel Encoder](#pixel-encoder)
	- [Estimators](#estimators)
		- [Anomaly Detectors](#anomaly-detectors)
			- [Isolation Forest](#isolation-forest)
			- [Isolation Tree](#isolation-tree)
			- [Local Outlier Factor](#local-outlier-factor)
			- [Robust Z Score](#robust-z-score)
		- [Classifiers](#classifiers)
			- [AdaBoost](#adaboost)
			- [Classification Tree](#classification-tree)
			- [Dummy Classifier](#dummy-classifier)
			- [Extra Tree](#extra-tree)
			- [Gaussian Naive Bayes](#gaussian-naive-bayes)
			- [K Nearest Neighbors](#k-nearest-neighbors)
			- [Logistic Regression](#logistic-regression)
			- [Multi Layer Perceptron](#multi-layer-perceptron)
			- [Naive Bayes](#naive-bayes)
			- [Random Forest](#random-forest)
			- [Softmax Classifier](#softmax-classifier)
		- [Clusterers](#clusterers)
			- [DBSCAN](#dbscan)
			- [Fuzzy C Means](#fuzzy-c-means)
			- [K Means](#k-means)
			- [Mean Shift](#mean-shift)
		- [Regressors](#regressors)
			- [Dummy Regressor](#dummy-regressor)
			- [KNN Regressor](#knn-regressor)
			- [MLP Regressor](#mlp-regressor)
			- [Regression Tree](#regression-tree)
			- [Ridge](#ridge)
	- [Estimator Interfaces](#estimator-interfaces)
		- [Online](#online)
		- [Probabilistic](#probabilistic)
		- [Persistable](#persistable)
	- [Meta-Estimators](#meta-estimators)
		- [Data Preprocessing](#data-preprocessing)
			- [Pipeline](#pipeline)
		- [Ensemble](#ensemble)
			- [Bootstrap Aggregator](#bootstrap-aggregator)
			- [Committee Machine](#committee-machine)
		- [Model Persistence](#model-persistence)
			- [Persistent Model](#persistent-model)
		- [Model Selection](#model-selection)
			- [Grid Search](#grid-search)
			- [Random Search](#random-search)
	- [Transformers](#transformers)
		- [Dense and Sparse Random Projectors](#dense-and-sparse-random-projectors)
		- [L1 and L2 Regularizers](#l1-and-l2-regularizers)
		- [Lambda Function](#lambda-function)
		- [Min Max Normalizer](#min-max-normalizer)
		- [Missing Data Imputer](#missing-data-imputer)
		- [Numeric String Converter](#numeric-string-converter)
		- [One Hot Encoder](#one-hot-encoder)
	    - [Polynomial Expander](#polynomial-expander)
	    - [Quartile Standardizer](#quartile-standardizer)
	    - [Robust Standardizer](#robust-standardizer)
		- [TF-IDF Transformer](#tf---idf-transformer)
		- [Variance Threshold Filter](#variance-threshold-filter)
		- [Z Scale Standardizer](#z-scale-standardizer)
	- [Neural Network](#neural-network)
		- [Activation Functions](#activation-functions)
			- [ELU](#elu)
			- [Hyperbolic Tangent](#hyperbolic-tangent)
			- [Identity](#identity)
			- [ISRU](#isru)
			- [Leaky ReLU](#leaky-relu)
			- [SELU](#selu)
			- [Sigmoid](#sigmoid)
			- [Soft Plus](#soft-plus)
			- [Softsign](#softsign)
		- [Layers](#layers)
			- [Input](#input)
			- [Hidden](#hidden)
				- [Dense](#dense)
			- [Output](#output)
				- [Linear](#linear)
				- [Logit](#logit)
				- [Softmax](#softmax)
		- [Optimizers](#optimizers)
			- [AdaGrad](#adagrad)
			- [Adam](#adam)
			- [Momentum](#momentum)
			- [RMS Prop](#rms-prop)
			- [Step Decay](#step-decay)
			- [Stochastic](#stochastic)
		- [Snapshots](#snapshots)
	- [Kernel Functions](#kernel-functions)
		- [Distance](#distance)
			- [Canberra](#canberra)
			- [Cosine](#cosine)
			- [Diagonal](#diagonal)
			- [Ellipsoidal](#ellipsoidal)
			- [Euclidean](#euclidean)
			- [Hamming](#hamming)
			- [Manhattan](#manhattan)
			- [Minkowski](#minkowski)
	- [Cross Validation](#cross-validation)
		- [Validators](#validators)
			- [Hold Out](#hold-out)
			- [K Fold](#k-fold)
			- [Leave P Out](#leave-p-out)
		- [Metrics](#validation-metrics)
			- [Anomaly Detection](#anomaly-detection)
			- [Classification](#classification)
			- [Clustering](#clustering)
			- [Regression](#regression)
	- [Reports](#reports)
		- [Aggregate Report](#aggregate-report)
		- [Confusion Matrix](#confusion-matrix)
		- [Contingency Table](#contingency-table)
		- [Multiclass Breakdown](#multiclass-breakdown)
		- [Outlier Ratio](#outlier-ratio)
		- [Prediction Speed](#prediction-speed)
		- [Residual Analysis](#residual-analysis)
	- [Other](#other)
		- [Guessing Strategies](#guessing-strategies)
			- [Blurry Mean](#blurry-mean)
			- [Blurry Median](#blurry-median)
			- [K Most Frequent](#k-most-frequent)
			- [Lottery](#lottery)
			- [Popularity Contest](#popularity-contest)
			- [Wild Guess](#wild-guess)
		- [Tokenizers](#tokenizers)
			- [Whitespace](#whitespace)
			- [Word](#word-tokenizer)

---
### An Introduction to Machine Learning in Rubix
Machine learning is the process by which a computer program is able to progressively improve performance on a certain task through training and data without explicitly being programmed. There are two types of learning that Rubix offers out of the box, *Supervised* and *Unsupervised*.
 - **Supervised** learning is a technique to train computer models with a dataset in which the outcome of each sample data point has been *labeled* either by a human expert or another ML model prior to training. There are two types of supervised learning to consider in Rubix:
	 - **Classification** is the problem of identifying which *class* a particular sample belongs to. For example, one task may be in determining a particular species of Iris flower based on its sepal and petal dimensions.
	 - **Regression** involves predicting continuous *values* rather than discrete classes. An example in which a regression model is appropriate would be predicting the life expectancy of a population based on economic factors.
- **Unsupervised** learning, by contrast, uses an *unlabeled* dataset and relies on the information within the training samples to learn insights.
	- **Clustering** is the process of grouping data points in such a way that members of the same group are more similar (homogeneous) than the rest of the samples. You can think of clustering as assigning a class label to an otherwise unlabeled sample. An example where clustering might be used is in differentiating tissues in PET scan images.
	- **Anomaly Detection** is the flagging of samples that do not conform to an expected pattern. Anomalous samples can often indicate adversarial activity, bad data, or exceptional performance.

### Obtaining Data
Machine learning projects typically begin with a question. For example, who of my friends are most likely to stay married to their spouse? One way to go about answering this question with machine learning would be to go out and ask a bunch of long-time married and divorced couples the same set of questions and then use that data to build a model of what a successful (or not) marriage looks like. Later, you can use that model to make predictions based on the answers from your friends.

Although this is certainly a valid way of obtaining data, in reality, chances are someone has already done the work of measuring the data for you and it is your job to find it, aggregate it, clean it, and otherwise make it usable by the machine learning algorithm. There are a number of PHP libraries out there that make extracting data from [CSV](https://github.com/thephpleague/csv), JSON, databases, and cloud services a whole lot easier, and we recommend checking them out before attempting it manually.

Having that said, Rubix will be able to handle any dataset as long as it can fit into one its predefined Dataset objects (Labeled, Unlabeled, etc.).

#### The Dataset Object
Data is passed around in Rubix via specialized data containers called Datasets. [Dataset objects](#dataset-objects) extend the PHP array structure with methods that properly handle selecting, splitting, folding, and randomizing the samples. In general, there are two types of Datasets, *Labeled* and *Unlabeled*. Labeled datasets are typically used for *supervised* learning and Unlabeled datasets are used for *unsupervised* learning and for making predictions.

For the following example, suppose that you went out and asked 100 couples (50 married and 50 divorced) to rate (between 1 and 5) their similarity, communication, and partner attractiveness. We could construct a [Labeled Dataset](#labeled) object from the data you collected in the following way:

```php
use \Rubix\ML\Datasets\Labeled;

$samples = [[3, 4, 2], [1, 5, 3], [4, 4, 3], [2, 1, 5], ...];

$labels = ['married', 'divorced', 'married', 'divorced', ...];

$dataset = new Labeled($samples, $labels);
```

The Dataset object is now ready to be used throughout Rubix.

### Choosing an Estimator

There are many different algorithms to chose from in Rubix and each one is designed to handle specific (sometimes overlapping) tasks. Choosing the right [Estimator](#estimators) for the job is crucial to building an accurate and performant computer model.

There are a couple ways that we could model our marriage satisfaction predictor. We could have asked a fourth question - that is, to rate each couple's overall marriage satisfaction from say 1 to 10 and then train a [Regressor](#regressors) to predict a continuous "satisfaction score" for each new sample. But since all we have to go by for now is whether they are still married or currently divorced, a [Classifier](#classifiers) will be better suited.

In practice, one will experiment with more than one type of Estimator to find the best fit to the data, but for the purposes of this introduction we will simply demonstrate a common and intuitive algorithm called *K Nearest Neighbors*.

#### Creating the Estimator Instance

Like most Estimators, the [K Nearest Neighbors](#k-nearest-neighbors) classifier requires a number of parameters (called *Hyperparameters*) to be chosen up front. These parameters can be selected based on some prior knowledge of the problem space, or at random. Rubix provides a meta-Estimator called [Grid Search](#grid-search) that searches the parameter space provided for the most effective combination. For the purposes of this example we will just go with our intuition and chose the parameters outright.

It is important to understand the effect that each parameter has on the performance of the Estimator since different settings can often lead to different results.

You can find a full description of all of the K Nearest Neighbors parameters in the [API reference](#api-reference) which we highly recommend reading a few times to get a grasp for what each parameter does.

The K Nearest Neighbors algorithm works by comparing the *distance* between a sample and each of the points from the training set. It will then use the K *nearest* points to base its prediction on. For example, if the 5 closest neighbors to a sample are 4 married and 1 divorced, the algorithm will output a prediction of married with a probability of 0.80.

To create a K Nearest Neighbors Classifier instance:
```php
use Rubix\ML\Classifiers\KNearestNeighbors;
use Rubix\ML\Kernels\Distance\Manhattan;

// Using the default parameters
$estimator = new KNearestNeighbors();

// Specifying parameters
$estimator = new KNearestNeighbors(3, new Manhattan());
```

Now that we've chosen and instantiated an Estimator and our Dataset object is ready to go, it is time to train our model and use it to make some predictions.

### Training and Prediction
Training is the process of feeding the Estimator data so that it can learn. The unique way in which the Estimator learns is based upon the underlying algorithm which has been implemented for you already. All you have to do is supply enough clean data so that the process can converge to a satisfactory optimum.

Passing the Labeled Dataset object we created earlier we can train the KNN estimator like so:
```php
...
$estimator->train($dataset);
```
That's it.

For our 100 sample dataset, this should only take a few microseconds, but larger datasets and more sophisticated Estimators can take much longer.

Once the Estimator has been fully trained we can feed in some new sample data points to see what the model predicts. Suppose that we went out and collected 5 new data points from our friends using the same questions we asked the couples we interviewed for our training set. We could make a prediction on whether they look more like the class of married or divorced couples by taking their answers and running them through the trained Estimator's `predict()` method.
```php
use Rubix\ML\Dataset\Unlabeled;

$samples = [[4, 1, 3], [2, 2, 1], [2, 4, 5], [5, 2, 4], [3, 2, 1];

$friends = new Unlabeled($samples);

$predictions = $estimator->predict($friends);

var_dump($predictions);
```

##### Output:

```sh
array(5) {
	[0] => 'divorced'
	[1] => 'divorced'
	[2] => 'married'
	[3] => 'married'
	[4] => 'divorced'
}
```
Note that we are not using a Labeled Dataset here because we don't know the outcomes yet. In fact, the label is exactly what we are trying to predict. Next we'll look at how we can test the accuracy of the predictions our model makes using cross validation.

### Evaluating Model Performance
Making predictions is not very useful unless the Estimator can correctly generalize what it has learned during training. [Cross Validation](#cross-validation) is a process by which we can test the model for its generalization ability. For the purposes of this introduction, we will use a simple form of cross validation called *Hold Out*. The [Hold Out](#hold-out) validator will take care of randomizing and splitting the dataset into  training and testing sets automatically, such that a portion of the data is *held out* to be used to test (or *validate*) the model. The reason we do not use *all* of the data for training is  because we want to test the Estimator on samples that it has never seen before.

Hold Out requires you to set the ratio of testing to training samples to use. In this case, let's chose to use a factor of 0.2 (20%) of the dataset for testing leaving the rest (80%) for training. Typically, 0.2 is a good default choice however your mileage may vary. The important thing to understand here is the trade off between more data for training and more precise testing results. Once you get the hang of Hold Out, the next step is to consider more elaborate cross validation techniques such as [K Fold](#k-fold), and [Leave P Out](#leave-p-out).

To return a validation score from the Hold Out Validator using the Accuracy Metric just pass it the untrained Estimator instance and a dataset.

##### Example:

```php
use Rubix\ML\CrossValidation\HoldOut;
use Rubix\ML\CrossValidation\Metrics\Accuracy;

...
$validator = new HoldOut(0.2);

$score = $validator->test($estimator, $dataset, new Accuracy());

var_dump($score);
```

##### Output:

```sh
float(0.945)
```

Since we are measuring accuracy, this output indicates that our Estimator is 94.5% accurate given the data we've trained and tested it with. Not bad.

### What Next?
Now that you've gone through a brief introduction of a simple machine learning problem in Rubix, the next step is to become more familiar with the API and to experiment with some data on your own. We highly recommend reading the entire documentation at least twice to fully understand all of the features at your disposal. If you're eager to get started, a great place to begin is by downloading some datasets from the University of California Irvine [Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets.html) where they have many pre-cleaned datasets available for free.

---
### API Reference
Here you will find information regarding the classes that make up the Rubix library.

### Datasets
Data is what powers machine learning programs so naturally we treat it as a first-class citizen. Rubix provides a number of classes that help you wrangle and even generate data.

### Dataset Objects
In Rubix, data is passed around using specialized data structures called Dataset objects. Dataset objects can hold a heterogeneous mix of categorical and ncontinuous data and gracefully handles *null* values with a user-defined placeholder. Dataset objects make it easy to slice and transport data in a canonical way.

There are two types of data that Estimators can process i.e. *categorical* and *continuous*. Any numerical (integer or float) datum is considered continuous and any string datum is considered categorical as a general rule throughout Rubix. This rule makes it easy to distinguish between the types of data while allowing for flexibility. For example, you could represent the number 5 as continuous by using the integer type or as categorical by using the string type (*'5'*).

##### Example:
```php
use Rubix\ML\Datasets\Unlabeled;

$samples = [
	['rough', 8, 6.55], ['furry', 10, 9.89], ...
];

$dataset = new Unlabeled($samples);
```

#### Selecting

Return the sample matrix:
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

Return the *first* **n** rows of data in a new Dataset object:
```php
public head(int $n = 10) : self
```

Return the *last* **n** rows of data in a new Dataset object:
```php
public tail(int $n = 10) : self
```

##### Example:
```php
// Return the sample matrix
$samples = $dataset->samples();

// Return just the first 5 rows in a new dataset
$subset = $dataset->head(5);
```

#### Properties

Return the number of rows in the Dataset:
```php
public numRows() : int
```

Return the number of columns in the Dataset:
```php
public numColumns() : int
```

#### Splitting, Folding, and Batching

Remove **n** rows from the Dataset and return them in a new Dataset:
```php
public take(int $n = 1) : self
```

Leave **n** samples on the Dataset and return the rest in a new Dataset:
```php
public leave(int $n = 1) : self
```

Split the Dataset into *left* and *right* subsets given by a **ratio**:
```php
public split(float $ratio = 0.5) : array
```

Fold the Dataset **k** - 1 times to form **k** equal size Datasets:
```php
public fold(int $k = 10) : array
```

Batch the Dataset into subsets of **n** rows per batch:
```php
public batch(int $n = 50) : array
```

##### Example:
```php
// Remove the first 5 rows and return them in a new dataset
$subset = $dataset->take(5);

// Split the dataset into left and right subsets
list($left, $right) = $dataset->split(0.5);

// Fold the dataset into 8 equal size datasets
$folds = $dataset->fold(8);
```

#### Randomizing

Randomize the order of the Dataset and return it:
```php
public randomize() : self
```
Generate a random subset of size **n**:
```php
public randomSubset($n = 1) : self
```
Generate a random subset with replacement of size **n**:
```php
public randomSubsetWithReplacement($n = 1) : self
```

##### Example:
```php
// Randomize and split the dataset into two subsets
list($left, $right) = $dataset->randomize()->split(0.8);

// Generate a dataset of 500 random samples
$subset = $dataset->randomSubset(500);
```

#### Sorting

To sort a Dataset by a specific feature column:
```php
public sortByColumn(int $index, bool $descending = false) : self
```

##### Example:
```php
...
var_dump($dataset->samples());

$dataset->sortByColumn(2, false);

var_dump($dataset->samples());
```

##### Output:
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

To prepend a given Dataset onto the beginning of another Dataset:
```php
public prepend(Dataset $dataset) : self
```

To append a given Dataset onto the end of another Dataset:
```php
public append(Dataset $dataset) : self
```

#### Applying a Transformation

You can apply a fitted [Transformer](#transformers) to a Dataset directly passing it to the apply method on the Dataset.

```php
public apply(Transformer $transformer) : void
```

##### Example:
```php
use Rubix\ML\Transformers\OneHotEncoder;

...
$transformer = new OneHotEncoder();

$transformer->fit($dataset);

$dataset->apply($transformer);
```

#### Saving and Restoring
Dataset objects can be saved and restored from a serialized object file which makes them easy to work with. Saving will capture the current state of the dataset including any transformations that have been applied.

Save the Dataset to a file:
```php
public save(?string $path = null) : void
```

Restore the Dataset from a file:
```php
public static restore(string $path) : self
```

##### Example:
```php
// Save the dataset to a file
$dataset->save('path/to/dataset');

// Assign a filename (ex. 1531772454.dataset)
$dataset->save();

$dataset = Labeled::restore('path/to/dataset');
```

There are two types of Dataset objects in Rubix, *labeled* and *unlabeled*.

### Labeled
For supervised Estimators you will need to pass it a Labeled Dataset consisting of a sample matrix and an array of labels that correspond to the observed outcomes of each sample. Splitting, folding, randomizing, sorting, and subsampling are all done while keeping the indices of samples and labels aligned.

In addition to the basic Dataset interface, the Labeled class can sort and *stratify* the data by label.

##### Parameters:
| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | samples | None | array | A 2-dimensional array consisting of rows of samples and columns of features. |
| 2 | labels | None | array | A 1-dimensional array of labels that correspond to the samples in the dataset. |
| 3| placeholder | '?' | mixed | The placeholder value for null features. |

##### Additional Methods:
Return a 1-dimensional array of labels:
```php
public labels() : array
```

Return the label at the given row offset:
```php
public label(int $index) : mixed
```

Return all of the possible outcomes i.e the unique labels:
```php
public possibleOutcomes() : array
```

Sort the Dataset by label:
```php
public sortByLabel(bool $descending = false) : self
```

Group the samples by label and return them in their own Dataset:
```php
public stratify() : array
```
Split the Dataset into left and right stratified subsets with a given **ratio** of samples in each:
```php
public stratifiedSplit($ratio = 0.5) : array
```

Fold the Dataset **k** - 1 times to form **k** equal size stratified Datasets
```php
public stratifiedFold($k = 10) : array
```

##### Example:
```php
use Rubix\ML\Datasets\Labeled;

...
$dataset = new Labeled($samples, $labels);

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

##### Output:
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

##### Example:
```php
...
// Fold the dataset into 5 equal size stratified subsets
$folds = $dataset->stratifiedFold(5);

// Split the dataset into two stratified subsets
list($left, $right) = $dataset->stratifiedSplit(0.8);

// Put each sample with label x into its own dataset
$strata = $dataset->stratify();
```

### Unlabeled
Unlabeled datasets can be used for training unsupervised Estimators and for feeding data into an Estimator to make predictions.

##### Parameters:
| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | samples | None | array | A 2-dimensional feature matrix consisting of rows of samples and columns of feature values. |
| 2| placeholder | '?' | mixed | The placeholder value for null features. |


##### Additional Methods:
This Dataset does not have any additional methods.

##### Example:
```php
use Rubix\ML\Datasets\Unlabeled;

$dataset = new Unlabeled($samples);
```

### Generators
Dataset Generators allow you to produce data of a user-specified shape, dimensionality, and cardinality. This is useful for augmenting a dataset with synthetic data or for testing and demonstration purposes.

To generate a Dataset object with **n** samples (*rows*):
```php
public generate(int $n = 100) : Dataset
```

Return the dimensionality of the samples produced:
```php
public dimensions() : int
```

##### Example:
```php
use Rubix\ML\Datasets\Generators\Blob;

$generator = new Blob([0, 0], 1.0);

$dataset = $generator->generate(3);

var_dump($generator->dimensions());

var_dump($dataset->samples());
```

##### Output:
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
An Agglomerate is a collection of other generators each given a label. Agglomerates are useful for classification, clustering, and anomaly detection problems where the label is a discrete value.

##### Parameters:
| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | generators | [ ] | array | A collection of generators keyed by their user-specified label (0 indexed by default). |
| 2 | weights | 1 / n | array | A set of arbitrary weight values corresponding to a generator's contribution to the agglomeration. |

##### Additional Methods:

Return the normalized weights of each generator in the agglomerate:
```php
public weights() : array
```

##### Example:
```php
use Rubix\ML\Datasets\Generators\Agglomerate;

$generator = new Agglomerate([
	new Blob([5, 2], 1.0),
	new HalfMoon([-3, 5], 1.5, 90.0, 0.1),
	new Circle([2, -4], 2.0, 0.05),
], [
	5, 6, 3, // Weights
]);
```

### Blob
A normally distributed n-dimensional blob of samples centered at a given mean vector. The standard deviation can be set for the whole blob or for each  feature column independently. When a global value is used, the resulting blob will be isotropic.

##### Parameters:
| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | center | [ ] | array | The coordinates of the center of the blob i.e. a centroid vector. |
| 2 | stddev | 1.0 | float or array | Either the global standard deviation or an array with the SD for each feature column. |

##### Additional Methods:
This Generator does not have any additional methods.

##### Example:
```php
use Rubix\ML\Datasets\Generators\Blob;

$generator = new Blob([1.2, 5.0, 2.6, 0.8], 0.25);
```

### Circle
Create a circle made of sample data points in 2 dimensions.

##### Parameters:
| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | center | [ ] | array | The x and y coordinates of the center of the circle. |
| 2 | scale | 1.0 | float | The scaling factor of the circle. |
| 3 | noise | 0.1 | float | The amount of Gaussian noise to add to each data point as a ratio of the scaling factor. |

##### Additional Methods:
This Generator does not have any additional methods.

##### Example:
```php
use Rubix\ML\Datasets\Generators\Circle;

$generator = new Circle([0.0, 0.0], 100, 0.1);
```

### Half Moon
Generate a dataset consisting of 2-d samples that form a half moon shape.

##### Parameters:
| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | center | [ ] | array | The x and y coordinates of the center of the circle. |
| 2 | scale | 1.0 | float | The scaling factor of the circle. |
| 3 | rotate | 90.0 | float | The amount in degrees to rotate the half moon counterclockwise. |
| 4 | noise | 0.1 | float | The amount of Gaussian noise to add to each data point as a ratio of the scaling factor. |

##### Additional Methods:
This Generator does not have any additional methods.

##### Example:
```php
use Rubix\ML\Datasets\Generators\HalfMoon;

$generator = new HalfMoon([0.0, 0.0], 100, 180.0, 0.2);
```

---
### Feature Extractors
Feature Extractors are objects that help you encode raw data into feature vectors so they can be used by an Estimator.

Extractors have an API similar to [Transformers](#transformers), however, they are designed to be used on the raw data *before* it is inserted into a Dataset Object. The output of the `extract()` method is a sample matrix that can be used to build a [Dataset Object](#dataset-objects).

Fit the Extractor to the raw samples before extracting:
```php
public fit(array $samples) : void
```

Return a sample matrix:
```php
public extract(array $samples) : array
```

##### Example:
```php
use Rubix\ML\Extractors\CountVectorizer;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Datasets\Labeled;

...
$estractor = new CountVectorizer(5000);

$extractor->fit($data);

$samples = $extractor->extract($data);

$dataset = new Unlabeled($samples);

$dataset = new Labeled($samples, $labels);
```

### Count Vectorizer
In machine learning, word *counts* are often used to represent natural language as numerical vectors. The Count Vectorizer builds a vocabulary from the training samples during fitting and transforms an array of strings (text *blobs*) into sparse feature vectors. Each feature column represents a word from the vocabulary and the value denotes the number of times that word appears in a given sample.

##### Parameters:
| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | max vocabulary | PHP_INT_MAX | int | The maximum number of words to encode into each word vector. |
| 2 | normalize | true | bool | Should we remove extra whitespace and lowercase? |
| 3 | tokenizer | Word | object | The object responsible for turning samples of text into individual tokens. |

##### Additional Methods:

Return the fitted vocabulary i.e. the words that will be vectorized:
```php
public vocabulary() : array
```

##### Example:
```php
use Rubix\ML\Extractors\CountVectorizer;
use Rubix\ML\Extractors\Tokenizers\Word;

$extractor = new CountVectorizer(5000, true, new Word());

// Return the vocabulary of the vectorizer
$extractor->vocabulary();

// Return the size of the fitted vocabulary
$extractor->size();
```

### Pixel Encoder
Images must first be converted to color channel values in order to be passed to an Estimator. The Pixel Encoder takes an array of images (as [PHP Resources](http://php.net/manual/en/language.types.resource.php)) and converts them to a flat vector of color channel data. Image scaling and cropping is handled automatically by [Intervention Image](http://image.intervention.io/). The GD extension is required to use this feature.

##### Parameters:
| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | size | [32, 32] | array | A tuple of width and height values denoting the resolution of the encoding. |
| 2 | rgb | true | bool | True to use RGB color channel data and False to use Greyscale. |
| 3 | sharpen | 0 | int | A value between 0 and 100 indicating the amount of sharpness to add to each sample. |
| 4 | driver | 'gd' | string | The PHP extension to use for image processing ('gd' *or* 'imagick'). |

##### Additional Methods:
This Extractor does not have any additional methods.

##### Example:
```php
use Rubix\ML\Extractors\PixelEncoder;

$extractor = new PixelEncoder([28, 28], false, 'imagick');
```

---
### Estimators
Estimators are the core of the Rubix library and consist of various [Classifiers](#classifiers), [Regressors](#regressors), [Clusterers](#clusterers), and [Anomaly Detectors](#anomaly-detectors) that make *predictions* based on their training. Estimators can be supervised or unsupervised depending on the task and can employ methods on top of the basic Estimator API by implementing a number of interfaces such as [Online](#online), [Probabilistic](#probabilistic), and [Persistable](#persistable). They can even be wrapped by a Meta-Estimator to provide additional functionality such as data [preprocessing](#pipeline) and [hyperparameter optimization](#grid-search).

To train an Estimator pass it a training Dataset:
```php
public train(Dataset $training) : void
```

To make predictions, pass it a new dataset:
```php
public predict(Dataset $dataset) : array
```

The return value of `predict()` is an array indexed in the order in which the samples were fed in.

##### Example:
```php
use Rubix\ML\Classifiers\RandomForest;
use Rubix\ML\Datasets\Labeled;

...
$dataset = new Labeled($samples, $labels);

$estimator = new RandomForest(200, 0.5, 5, 3);

// Take 3 samples out of the dataset to use later
$testing = $dataset->take(3);

// Train the estimator with the labeled dataset
$estimator->train($dataset);

// Make some predictions on the holdout set
$result = $estimator->predict($testing);

var_dump($result);
```

##### Output:
```sh
array(3) {
	[0] => 'married'
	[1] => 'divorced'
	[2] => 'married'
}
```

---
### Anomaly Detectors

[Anomaly detection](https://en.wikipedia.org/wiki/Anomaly_detection) is the process of identifying samples that do not conform to an expected pattern. They can be used in fraud prevention, intrusion detection, sciences, and many other areas. The output of a Detector is either *0* for a normal sample or *1* for a detected anomaly.

### Isolation Forest
An [Ensemble](#ensemble) Anomaly Detector comprised of [Isolation Trees](#isolation-tree) each trained on a different subset of the training set. The Isolation Forest works by averaging the isolation score of a sample across a user-specified number of trees.

##### Unsupervised | Probabilistic | Persistable | Nonlinear

##### Parameters:
| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | trees | 100 | int | The number of Isolation Trees to train in the ensemble. |
| 2 | ratio | 0.1 | float | The ratio of random samples to train each Isolation Tree with. |
| 3 | threshold | 0.5 | float | The threshold isolation score. i.e. the probability that a sample is an outlier. |

##### Additional Methods:
This Estimator does not have any additional methods.

##### Example:
```php
use Rubix\ML\AnomalyDetection\IsolationForest;

$estimator = new IsolationForest(500, 0.1, 0.7);
```
### Isolation Tree
Isolation Trees separate anomalous samples from dense clusters using an extremely randomized splitting process that isolates outliers into their own nodes. *Note* that this Estimator is considered a *weak* learner and is typically used within the context of an ensemble (such as [Isolation Forest](#isolation-forest)).

##### Unsupervised | Probabilistic | Persistable | Nonlinear

##### Parameters:
| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | max depth | PHP_INT_MAX | int | The maximum depth of a branch that is allowed. |
| 2 | threshold | 0.5 | float | The minimum isolation score. i.e. the probability that a sample is an outlier. |

##### Additional Methods:
This Estimator does not have any additional methods.

##### Example:
```php
use Rubix\ML\AnomalyDetection\IsolationTree;

$estimator = new IsolationTree(1000, 0.65);
```

### Local Outlier Factor
The Local Outlier Factor (LOF) algorithm considers the local region of a sample, set by the k parameter, when determining an outlier. A density estimate for each neighbor is computed by measuring the radius of the cluster centroid that the point and its neighbors form. The LOF is the ratio of the sample over the median radius of the local region.

##### Unsupervised | Probabilistic | Online | Persistable | Nonlinear

##### Parameters:
| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | k | 10 | int | The k nearest neighbors that form a local region. |
| 2 | neighbors | 20 | int | The number of neighbors considered when computing the radius of a centroid. |
| 3 | threshold | 0.5 | float | The minimum density score. i.e. the probability that a sample is an outlier. |
| 4 | kernel | Euclidean | object | The distance metric used to measure the distance between two sample points. |

##### Additional Methods:
This Estimator does not have any additional methods.

##### Example:
```php
use Rubix\ML\AnomalyDetection\LocalOutlierFactor;
use Rubix\ML\Kernels\Distance\Minkowski;

$estimator = new LocalOutlierFactor(10, 20, 0.2, new Minkowski(3.5));
```

### Robust Z Score
A quick *global* anomaly detector, Robust Z Score uses a modified Z score to detect outliers within a Dataset. The modified Z score consists of taking the median and median absolute deviation (MAD) instead of the mean and standard deviation thus making the statistic more robust to training sets that may already contain outliers. Outlier can be flagged in one of two ways. First, their average Z score can be above the user-defined tolerance level or an individual feature's score could be above the threshold (*hard* limit).

##### Unsupervised | Persistable

##### Parameters:
| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | tolerance | 3.0 | float | The average z score to tolerate before a sample is considered an outlier. |
| 2 | threshold | 3.5 | float | The threshold z score of a individual feature to consider the entire sample an outlier. |

##### Additional Methods:

Return the median of each feature column in the training set:
```php
public medians() : array
```

Return the median absolute deviation (MAD) of each feature column in the training set:
```php
public mads() : array
```

##### Example:
```php
use Rubix\ML\AnomalyDetection\RobustZScore;

$estimator = new RobustZScore(1.5, 3.0);
```

---
### Classifiers
Classifiers are a type of Estimator that predict discrete outcomes such as class labels. There are two types of Classifiers in Rubix - *Binary* and *Multiclass*. Binary Classifiers can only distinguish between two classes (ex. *Male*/*Female*, *Yes*/*No*, etc.) whereas a Multiclass Classifier is able to handle two or more unique class outcomes.

### AdaBoost
Short for Adaptive Boosting, this ensemble classifier can improve the performance of an otherwise *weak* classifier by focusing more attention on samples that are harder to classify.

##### Supervised | Binary | Persistable

##### Parameters:
| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | base | None | string | The fully qualified class name of the base *weak* classifier. |
| 2 | params | [ ] | array | The parameters of the base classifer. |
| 3 | estimators | 100 | int | The number of estimators to train in the ensemble. |
| 4 | ratio | 0.1 | float | The ratio of samples to subsample from the training dataset per epoch. |
| 5 | tolerance | 1e-3 | float | The amount of validation error to tolerate before an early stop is considered. |

##### Additional Methods:

Return the calculated weight values of the last trained dataset:
```php
public weights() : array
```

Return the influence scores for each boosted classifier:
```php
public influence() : array
```

##### Example:
```php
use Rubix\ML\Classifiers\AdaBoost;
use Rubix\ML\Classifiers\ExtraTree;

$estimator = new AdaBoost(ExtraTree::class, [10, 3, 5], 200, 0.1, 1e-2);
```

### Classification Tree
A Decision Tree-based classifier that minimizes [gini impurity](https://en.wikipedia.org/wiki/Gini_coefficient) to greedily search for the best splits in a training set.

##### Supervised | Multiclass | Probabilistic | Persistable | Nonlinear

##### Parameters:
| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | max depth | PHP_INT_MAX | int | The maximum depth of a branch that is allowed. Setting this to 1 is equivalent to training a Decision Stump. |
| 2 | min samples | 5 | int | The minimum number of data points needed to make a prediction. |
| 3 | max features | PHP_INT_MAX | int | The maximum number of features to consider when determining a split point. |
| 4 | tolerance | 1e-3 | float | A small amount of impurity to tolerate when choosing a split. |

##### Additional Methods:
This Estimator does not have any additional methods.

##### Example:
```php
use Rubix\ML\Classifiers\ClassificationTree;

$estimator = new ClassificationTree(100, 7, 4, 1e-4);
```


### Dummy Classifier
A classifier that uses a user-defined [Guessing Strategy](#guessing-strategies) to make predictions. Dummy Classifier is useful to provide a sanity check and to compare performance with an actual classifier.

##### Supervised | Multiclass | Persistable

##### Parameters:
| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | strategy | PopularityContest | object | The guessing strategy to employ when guessing the outcome of a sample. |

##### Additional Methods:
This Estimator does not have any additional methods.

##### Example:
```php
use Rubix\ML\Classifiers\DummyClassifier;
use Rubix\ML\Other\Strategies\PopularityContest;

$estimator = new DummyClassifier(new PopularityContest());
```

### Extra Tree
An Extremely Randomized Classification Tree that splits the training set at a random point chosen among the maximum features. Extra Trees work great in Ensembles such as [Random Forest](#random-forest) or [AdaBoost](#adaboost) as the "weak" classifier or they can be used on their own. The strength of Extra Trees are computational efficiency as well as increasing variance of the prediction (if that is desired).

##### Supervised | Multiclass | Probabilistic | Persistable | Nonlinear

##### Parameters:
| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | max depth | PHP_INT_MAX | int | The maximum depth of a branch that is allowed. Setting this to 1 is equivalent to training a Decision Stump. |
| 2 | min samples | 5 | int | The minimum number of data points needed to make a prediction. |
| 3 | max features | PHP_INT_MAX | int | The number of features to consider when determining a split. |

##### Additional Methods:
This Estimator does not have any additional methods.

##### Example:
```php
use Rubix\ML\Classifiers\ExtraTree;

$estimator = new ExtraTree(20, 3, 4);
```

### Gaussian Naive Bayes
A variate of the [Naive Bayes](#naive-bayes) classifier that uses a probability density function (*PDF*) over continuous features. The distribution of values is assumed to be Gaussian therefore your data might need to be transformed beforehand if it is not normally distributed.

##### Supervised | Multiclass | Online | Probabilistic | Persistable | Nonlinear

##### Parameters:
This Estimator does not have any parameters.

##### Additional Methods:

Return the class prior log probabilities based on their weight over all training samples:
```php
public priors() : array
```

Return the running mean of each feature column of the training data:
```php
public means() : array
```

Return the running variance of each feature column of the training data:
```php
public variances() : array
```

##### Example:
```php
use Rubix\ML\Classifiers\GaussianNB;

$estimator = new GaussianNB();
```

### K Nearest Neighbors
A distance-based algorithm that locates the K nearest neighbors from the training set and uses a majority vote to classify the unknown sample. K Nearest Neighbors is considered a *lazy* learning Estimator because it does the majority of its computation at prediction time.

##### Supervised | Multiclass | Probabilistic | Online | Persistable | Nonlinear

##### Parameters:
| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | k | 5 | int | The number of neighboring training samples to consider when making a prediction. |
| 2 | kernel | Euclidean | object | The distance kernel used to measure the distance between two sample points. |

##### Additional Methods:
This Estimator does not have any additional methods.

##### Example:
```php
use Rubix\ML\Classifiers\KNearestNeighbors;
use Rubix\ML\Kernels\Distance\Euclidean;

$estimator = new KNearestNeighbors(3, new Euclidean());
```

### Logistic Regression
A type of linear classifier that uses the logistic (sigmoid) function to distinguish between two possible outcomes. Logistic Regression measures the relationship between the class label and one or more independent variables by estimating probabilities.

##### Supervised | Binary | Online | Probabilistic | Persistable | Linear

##### Parameters:
| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | epochs | 100 | int | The maximum number of training epochs to execute. |
| 2 | batch size | 10 | int | The number of training samples to process at a time. |
| 3 | optimizer | Adam | object | The gradient descent optimizer used to train the underlying network. |
| 4 | alpha | 1e-4 | float | The L2 regularization term. |
| 5 | min change | 1e-8 | float | The minimum change in the weights necessary to continue training. |

##### Additional Methods:
Return the training progress at each epoch:
```php
public progress() : array
```

Return the underlying neural network instance or *null* if untrained:
```php
public network() : Network|null
```

##### Example:
```php
use Rubix\ML\Classifers\LogisticRegression;
use Rubix\ML\NeuralNet\Optimizers\Adam;

$estimator = new LogisticRegression(300, 10, new Adam(0.001), 1e-4, 1e-8);
```

### Multi Layer Perceptron
A multiclass feedforward [Neural Network](#neural-network) classifier that uses a series of user-defined [Hidden Layers](#hidden) as intermediate computational units. Multiple layers and non-linear activation functions allow the Multi Layer Perceptron to handle complex non-linear problems. MLP also features progress monitoring which stops training when it can no longer make progress. It also utilizes [snapshotting](#snapshots) to make sure that it always uses the best parameters even if progress may have declined during training.

##### Supervised | Multiclass | Online | Probabilistic | Persistable | Nonlinear

##### Parameters:
| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | hidden | [ ] | array | An array of hidden layers of the neural network. |
| 2 | batch size | 50 | int | The number of training samples to process at a time. |
| 3 | optimizer | Adam | object | The gradient descent step optimizer used to train the underlying network. |
| 4 | alpha | 1e-4 | float | The L2 regularization term. |
| 5 | metric | Accuracy | object | The validation metric used to monitor the training progress of the network. |
| 6 | holdout | 0.1 | float | The ratio of samples to hold out for progress monitoring. |
| 7 | window | 3 | int | The number of epochs to consider when determining if the algorithm should terminate or keep training. |
| 8 | tolerance | 1e-3 | The amount of error to tolerate in the validation metric. |
| 9 | epochs | PHP_INT_MAX | int | The maximum number of training epochs to execute. |

##### Additional Methods:
Return the training progress at each epoch:
```php
public progress() : array
```

Returns the underlying neural network instance or *null* if untrained:
```php
public network() : Network|null
```

##### Example:
```php
use Rubix\ML\Classifiers\MultiLayerPerceptron;
use Rubix\ML\NeuralNet\Layers\Dense;
use Rubix\ML\NeuralNet\ActivationFunctions\ELU;
use Rubix\ML\NeuralNet\Optimizers\Adam;
use Rubix\ML\CrossValidation\Metrics\MCC;

$hidden = [
	new Dense(30, new ELU()),
	new Dense(20, new ELU()),
	new Dense(10, new ELU()),
];

$estimator = new MultiLayerPerceptron($hidden, 100, new Adam(0.001), 1e-4, new MCC(), 0.2, 3, PHP_INT_MAX);
```

### Naive Bayes
Probability-based classifier that used probabilistic inference to predict the correct class. The probabilities are calculated using [Bayes Rule](https://en.wikipedia.org/wiki/Bayes%27_theorem). The naive part relates to the fact that it assumes that all features are *independent*, which is rarely the case in the real world but tends to work out in practice for most problems.

##### Supervised | Multiclass | Online | Probabilistic | Persistable | Nonlinear

##### Parameters:
| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | smoothing | 1.0 | float | The amount of additive (Laplace) smoothing to apply to the probabilities. |

##### Additional Methods:

Returns the class prior log probabilities based on their weight over all training samples:
```php
public priors() : array
```

Return the log probabilities of each feature given each class label:
```php
public probabilities() : array
```

##### Example:
```php
use Rubix\ML\Classifiers\NaiveBayes;

$estimator = new NaiveBayes(10.0);
```

### Random Forest
[Ensemble](#ensemble) classifier that trains Decision Trees ([Classification Trees](#classification-tree) or [Extra Trees](#extra-tree)) on a random subset (*bootstrap*) of the training data. A prediction is made based on the probability scores returned from each tree in the forest weighted equally.

##### Supervised | Multiclass | Probabilistic | Persistable | Nonlinear

##### Parameters:
| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | trees | 100 | int | The number of Decision Trees to train in the ensemble. |
| 2 | ratio | 0.1 | float | The ratio of random samples to train each Decision Tree with. |
| 3 | max depth | 10 | int | The maximum depth of a branch that is allowed. Setting this to 1 is equivalent to training a Decision Stump. |
| 4 | min samples | 5 | int | The minimum number of data points needed to split a decision node. |
| 5 | max features | PHP_INT_MAX | int | The number of features to consider when determining a split. |
| 6 | tolerance | 1e-3 | float | A small amount of Gini impurity to tolerate when choosing a split. |
| 7 | base | ClassificationTree::class | string | The base tree class name. |

##### Additional Methods:
This Estimator does not have any additional methods.

##### Example:
```php
use Rubix\ML\Classifiers\RandomForest;
use Rubix\ML\Classifiers\ClassificationTree;

$estimator = new RandomForest(400, 0.1, 10, 3, 5, 1e-2, ClassificationTree::class);
```

### Softmax Classifier
A generalization of [Logistic Regression](#logistic-regression) for multiclass problems using a single layer [neural network](#neural-network) with a Softmax output layer.

##### Supervised | Multiclass | Online | Probabilistic | Persistable | Linear

##### Parameters:
| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | epochs | 100 | int | The maximum number of training epochs to execute. |
| 2 | batch size | 10 | int | The number of training samples to process at a time. |
| 3 | optimizer | Adam | object | The gradient descent step optimizer used to train the underlying network. |
| 4 | alpha | 1e-4 | float | The L2 regularization term. |
| 5 | min change | 1e-8 | float | The minimum change in the weights necessary to continue training. |

##### Additional Methods:

Return the training progress at each epoch:
```php
public progress() : array
```

Return the underlying neural network instance or *null* if untrained:
```php
public network() : Network|null
```

##### Example:
```php
use Rubix\ML\Classifiers\SoftmaxClassifier;
use Rubix\ML\NeuralNet\Optimizers\Momentum;

$estimator = new SoftmaxClassifier(300, 100, new Momentum(0.001), 1e-4, 1e-5);
```

---
### Clusterers
Clustering is a common technique in machine learning that focuses on grouping samples in such a way that the groups are similar. Clusterers take unlabeled data points and assign them a label (cluster). The return value of each prediction is the cluster number each sample was assigned to.

### DBSCAN
Density-Based Spatial Clustering of Applications with Noise is a clustering algorithm able to find non-linearly separable and arbitrarily-shaped clusters. In addition, DBSCAN also has the ability to mark outliers as *noise* and thus can be used as a quasi [Anomaly Detector](#anomaly-detectors) as well.

##### Unsupervised | Persistable | Nonlinear

##### Parameters:
| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | radius | 0.5 | float | The maximum radius between two points for them to be considered in the same cluster. |
| 2 | min density | 5 | int | The minimum number of points within radius of each other to form a cluster. |
| 3 | kernel | Euclidean | object | The distance metric used to measure the distance between two sample points.

##### Additional Methods:
This Estimator does not have any additional methods.

##### Example:
```php
use Rubix\ML\Clusterers\DBSCAN;
use Rubix\ML\Kernels\Distance\Diagonal;

$estimator = new DBSCAN(4.0, 5, new Diagonal());
```

### Fuzzy C Means
Distance-based clusterer that allows samples to belong to multiple clusters if they fall within a fuzzy region defined by the fuzz parameter. Fuzzy C Means is similar to both K Means and Gaussian Mixture Models in that they require apriori knowledge of the number (parameter *c*) of clusters.

##### Unsupervised | Probabilistic | Persistable

##### Parameters:
| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | c | None | int | The number of target clusters. |
| 2 | fuzz | 2.0 | float | Determines the bandwidth of the fuzzy area. |
| 3 | kernel | Euclidean | object | The distance metric used to measure the distance between two sample points. |
| 4 | threshold | 1e-4 | float | The minimum change in centroid means necessary for the algorithm to continue training. |
| 5 | epochs | PHP_INT_MAX | int | The maximum number of training rounds to execute. |

##### Additional Methods:

Return the *c* computed centroids of the training data:
```php
public centroids() : array
```

Returns the progress of training at each epoch:
```php
public progress() : array
```

##### Example:
```php
use Rubix\ML\Clusterers\FuzzyCMeans;
use Rubix\ML\Kernels\Distance\Euclidean;

$estimator = new FuzzyCMeans(5, 1.2, new Euclidean(), 1e-3, 1000);
```

### K Means
A fast centroid-based hard clustering algorithm capable of clustering linearly separable data points given a number of target clusters set by the parameter K.

##### Unsupervised | Online | Persistable | Linear

##### Parameters:
| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | k | None | int | The number of target clusters. |
| 2 | kernel | Euclidean | object | The distance metric used to measure the distance between two sample points. |
| 3 | epochs | PHP_INT_MAX | int | The maximum number of training rounds to execute. |

##### Additional Methods:

Return the *c* computed centroids of the training data:
```php
public centroids() : array
```

##### Example:
```php
use Rubix\ML\Clusterers\KMeans;
use Rubix\ML\Kernels\Distance\Euclidean;

$estimator = new KMeans(3, new Euclidean());
```

### Mean Shift
A hierarchical clustering algorithm that uses peak finding to locate the local maxima (centroids) of a training set given by a radius constraint.

##### Unsupervised | Persistable | Linear

##### Parameters:
| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | radius | None | float | The radius of each cluster centroid. |
| 2 | kernel | Euclidean | object | The distance metric used to measure the distance between two sample points. |
| 3 | threshold | 1e-8 | float | The minimum change in centroid means necessary for the algorithm to continue training. |
| 4 | epochs | PHP_INT_MAX | int | The maximum number of training rounds to execute. |


##### Additional Methods:

Return the *c* computed centroids of the training data:
```php
public centroids() : array
```

Returns the progress of training at each epoch:
```php
public progress() : array
```

##### Example:
```php
use Rubix\ML\Clusterers\MeanShift;
use Rubix\ML\Kernels\Distance\Diagonal;

$estimator = new MeanShift(3.0, new Diagonal(), 1e-6, 2000);
```

---
### Regressors
Regression analysis is used to predict the outcome of an event where the value is continuous.

### Dummy Regressor
Regressor that guesses the output values based on a [Guessing Strategy](#guessing-strategies). Dummy Regressor is useful to provide a sanity check and to compare performance against actual Regressors.

##### Supervised | Persistable

##### Parameters:
| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | strategy | BlurryMean | object | The guessing strategy to employ when guessing the outcome of a sample. |

##### Additional Methods:
This Estimator does not have any additional methods.

##### Example:
```php
use Rubix\ML\Regressors\DummyRegressor;
use Rubix\ML\Other\Strategies\BlurryMedian;

$estimator = new DummyRegressor(new BlurryMedian(0.2));
```

### KNN Regressor
A version of [K Nearest Neighbors](#k-nearest-neighbors) that uses the mean outcome of K nearest data points to make continuous valued predictions suitable for regression problems.

##### Supervised | Online | Persistable | Nonlinear

##### Parameters:
| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | k | 5 | int | The number of neighboring training samples to consider when making a prediction. |
| 2 | kernel | Euclidean | object | The distance kernel used to measure the distance between two sample points. |

##### Additional Methods:
This Estimator does not have any additional methods.

##### Example:
```php
use Rubix\ML\Regressors\KNNRegressor;
use Rubix\ML\Kernels\Distance\Minkowski;

$estimator = new KNNRegressor(2, new Minkowski(3.0));
```

### MLP Regressor
A [Neural Network](#neural-network) with a linear output layer suitable for regression problems. The MLP also features progress monitoring which stops training when it can no longer make progress. It also utilizes [snapshotting](#snapshots) to make sure that it always uses the best parameters even if progress declined during training.

##### Supervised | Persistable | Nonlinear

##### Parameters:
| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | hidden | [ ] | array | An array of hidden layers of the neural network. |
| 2 | batch size | 10 | int | The number of training samples to process at a time. |
| 3 | optimizer | Adam | object | The gradient descent step optimizer used to train the underlying network. |
| 4 | alpha | 1e-4 | float | The L2 regularization term. |
| 5 | metric | Accuracy | object | The validation metric used to monitor the training progress of the network. |
| 6 | holdout | 0.1 | float | The ratio of samples to hold out for progress monitoring. |
| 7 | window | 3 | int | The number of epochs to consider when determining if the algorithm should terminate or keep training. |
| 8 | tolerance | float | 1e-5 | The amount of error to tolerate in the validation metric. |
| 9 | epochs | PHP_INT_MAX | int | The maximum number of training epochs to execute. |


##### Additional Methods:

Return the training progress at each epoch:
```php
public progress() : array
```

Returns the underlying neural network instance or *null* if untrained:
```php
public network() : Network|null
```

##### Example:
```php
use Rubix\ML\Regressors\MLPRegressor;
use Rubix\ML\NeuralNet\Layers\Dense;
use Rubix\ML\NeuralNet\ActivationFunctions\ISRU;
use Rubix\ML\NeuralNet\Optimizers\RMSProp;
use Rubix\ML\CrossValidation\Metrics\MeanSquaredError;

$hidden = [
	new Dense(100, new ISRU()),
];

$estimator = new MLPRegressor($hidden, 50, new RMSProp(0.001), 1e-2, new MeanSquaredError(), 0.1, 3, PHP_INT_MAX);
```

### Regression Tree
A Decision Tree learning algorithm that performs greedy splitting by minimizing the sum of squared errors between decision node splits.

##### Supervised | Persistable | Nonlinear

##### Parameters:
| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | max depth | PHP_INT_MAX | int | The maximum depth of a branch that is allowed. |
| 2 | min samples | 5 | int | The minimum number of data points needed to make a prediction. |
| 3 | max features | PHP_INT_MAX | int | The maximum number of features to consider when determining a split point. |
| 4 | tolerance | 1e-4 | float | A small amount of impurity to tolerate when choosing a split. |

##### Additional Methods:
This Estimator does not have any additional methods.

##### Example:
```php
use Rubix\ML\Regressors\RegressionTree;

$estimator = new RegressionTree(80, 1, 10, 0.0);
```

### Ridge
L2 penalized least squares linear regression. Can be used for simple regression problems that can be fit to a straight line.

##### Supervised | Persistable | Linear

##### Parameters:
| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | alpha | 1.0 | float | The L2 regularization term. |

##### Additional Methods:

Return the y intercept of the computed regression line:
```php
public intercept() : float|null
```

Return the computed coefficients of the regression line:
```php
public coefficients() : array
```

##### Example:
```php
use Rubix\ML\Regressors\Ridge;

$estimator = new Ridge(2.0);
```

---
### Estimator Interfaces

### Online

Certain [Estimators](#estimators) that implement the *Online* interface can be trained in batches. Estimators of this type are great for when you either have a continuous stream of data or a dataset that is too large to fit into memory. Partial training allows the model to grow as new data is acquired.

You can partially train an Online Estimator with:
```php
public partial(Dataset $dataset) : void
```

##### Example:
```php
...
$datasets = $dataset->fold(3);

$estimator->partial($dataset[0]);

$estimator->partial($dataset[1]);

$estimator->partial($dataset[2]);
```

It is *important* to note that an Estimator will continue to train as long as you are using the `partial()` method, however, calling `train()` on a trained or partially trained Estimator will reset it back to baseline first.

---
### Probabilistic

Some [Estimators](#estimators) may implement the *Probabilistic* interface, in which case, they will have an additional method that returns an array of probability scores of each possible class, cluster, etc. Probabilities are useful for ascertaining the degree to which the Estimator is certain about a particular outcome.

Calculate probability estimates:
```php
public proba(Dataset $dataset) : array
```

##### Example:
```php
...
$probabilities = $estimator->proba($dataset->head(2));  

var_dump($probabilities);
```

##### Output:
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

---
### Meta-Estimators
Meta-Estimators allow you to progressively enhance your models by adding additional functionality such as [data preprocessing](#data-preprocessing) and [persistence](#model-persistence) or by orchestrating an [Ensemble](#ensemble) of base Estimators. Each Meta-Estimator wraps a base Estimator and you can even wrap certain Meta-Estimators with other Meta-Estimators. Some examples of Meta-Estimators in Rubix are [Pipeline](#pipeline), [Grid Search](#grid-search), and [Bootstrap Aggregator](#bootstrap-aggregator).

##### Example:
```php
use Rubix\ML\Pipeline;
use Rubix\ML\GridSearch;
use Rubix\ML\Classifiers\ClassificationTree;
use Rubix\ML\CrossValidation\Metrics\MCC;
use Rubix\ML\CrossValidation\KFold;
use Rubix\ML\Transformers\NumericStringConverter;

...
$params = [[10, 30, 50], [1, 3, 5], [2, 3, 4];

$estimator = new Pipeline(new GridSearch(ClassificationTree::class, $params, new MCC(), new KFold(10)));

$estimator->train($dataset); // Train a classification tree with preprocessing and grid search

$estimator->complexity(); // Call complexity() method on Decision Tree from Pipeline
```
---
### Data Preprocessing
Often, additional processing of input data is required to deliver correct predictions and/or accelerate the training process. In this section, we'll introduce the Pipeline meta-Estimator and the various [Transformers](#transformers) that it employs to fit the input data to suit the requirements and preferences of the [Estimator](#estimator) that it feeds.

### Pipeline
Pipeline is responsible for transforming the input sample matrix of a Dataset in such a way that can be processed by the base Estimator. Pipeline accepts a base Estimator and a list of Transformers to apply to the input data before it is fed to the learning algorithm. Under the hood, Pipeline will automatically fit the training set upon training and transform any [Dataset object](#dataset-objects) supplied as an argument to one of the base Estimator's methods, including `predict()`.

##### Classifiers, Regressors, Clusterers, Anomaly Detectors

##### Parameters:
| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | estimator | None | object | An instance of a base estimator. |
| 2 | transformers | [ ] | array | The transformer middleware to be applied to each dataset. |

##### Additional Methods:
This Meta Estimator does not have any additional methods.

##### Example:
```php
use Rubix\ML\Pipeline;
use Rubix\ML\Classifiers\SoftmaxClassifier;
use Rubix\ML\NeuralNet\Optimizer\RMSProp;
use Rubix\ML\Transformers\MissingDataImputer;
use Rubix\ML\Transformers\OneHotEncoder;
use Rubix\ML\Transformers\SparseRandomProjector;

$estimator = new Pipeline(new SoftmaxClassifier(100, new RMSProp(0.01), 1e-2), [
	new MissingDataImputer(),
	new OneHotEncoder(),
	new SparseRandomProjector(30),
]);

$estimator->train($dataset); // Datasets are fit and ...

$estimator->predict($samples); // Transformed automatically.
```

Transformer *middleware* will process in the order given when the Pipeline was built and cannot be reordered without instantiating a new one. Since Tranformers run sequentially, the order in which they run *matters*. For example, a Transformer near the end of the stack may depend on a previous Transformer to convert all categorical features into continuous ones before it can run.

In practice, applying transformations can drastically improve the performance of your model by cleaning, scaling, expanding, compressing, and normalizing the input data.

---
### Ensemble
Ensemble Meta Estimators train and orchestrate a number of base Estimators in order to make their predictions. Certain Estimators (like [AdaBoost](#adaboost) and [Random Forest](#random-forest)) are implemented as Ensembles under the hood, however these *Meta* Estimators are able to work across Estimator types which makes them very useful.

### Bootstrap Aggregator
Bootstrap Aggregating (or *bagging*) is a model averaging technique designed to improve the stability and performance of a user-specified base Estimator by training a number of them on a unique bootstrapped training set. Bootstrap Aggregator then collects all of their predictions and makes a final prediction based on the results.

##### Classifiers, Regressors, Anomaly Detectors

##### Parameters:
| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | base | None | string | The fully qualified class name of the base Estimator. |
| 2 | params | [ ] | array | The parameters of the base estimator. |
| 3 | estimators | 10 | int | The number of base estimators to train in the ensemble. |
| 4 | ratio | 0.5 | float | The ratio of random samples to train each estimator with. |

##### Additional Methods:
This Meta Estimator does not have any additional methods.

##### Example:
```php
use Rubix\ML\BootstrapAggregator;
use Rubix\ML\Regressors\RegressionTree;

...
$estimator = new BootstrapAggregator(RegressionTree::class, [10, 5, 3], 100, 0.2);

$estimator->traing($training); // Trains 100 regression trees

$estimator->predict($testing); // Aggregates their predictions
```

### Committee Machine
A voting Ensemble that aggregates the predictions of a committee of user-specified, heterogeneous estimators (called *experts*) of a single type (i.e all Classifiers, Regressors, etc). The committee uses a hard-voting scheme to make final predictions.

##### Classifiers, Regressors, Anomaly Detectors

##### Parameters:
| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | experts | [ ] | array | An array of estimator instances. |


##### Additional Methods:
This Meta Estimator does not have any additional methods.

##### Example:
```php
use Rubix\ML\Classifiers\CommitteeMachine;
use Rubix\ML\Classifiers\RandomForest;
use Rubix\ML\Classifiers\SoftmaxClassifier;
use Rubix\ML\NeuralNet\Optimizers\Adam;
use Rubix\ML\Classifiers\KNearestNeighbors;

$estimator = new CommitteeMachine([
	new RandomForest(100, 0.3, 30, 3, 4, 1e-3),
	new SoftmaxClassifier(50, new Adam(0.001), 0.1),
	new KNearestNeighbors(3),
]);
```

---
### Model Selection
Model selection is the task of selecting a version of a model with a hyperparameter combination that maximizes performance on a specific validation metric. Rubix provides the *Grid Search* meta-Estimator that performs an exhaustive search over all combinations of parameters given as possible arguments.

### Grid Search
Grid Search is an algorithm that optimizes hyperparameter selection. From the user's perspective, the process of training and predicting is the same, however, under the hood, Grid Search trains one [Estimator](#estimators) per combination of parameters and predictions are made using the best Estimator. You can access the scores for each parameter combination by calling the `results()` method on the trained Grid Search meta-Estimator or you can get the best parameters by calling `best()`.

##### Parameters:
| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | base | None | string | The fully qualified class name of the base Estimator. |
| 2 | params | [ ] | array | An array containing n-tuples of parameters where each tuple represents a possible parameter for a given parameter location (ordinal). |
| 3 | metric | None | object | The validation metric used to score each set of parameters. |
| 4 | validator | None | object | An instance of a Validator object (HoldOut, KFold, etc.) that will be used to test each parameter combination. |

##### Additional Methods:

Return the results (scores and parameters) of the last search:
```php
public results() : array
```

Return the he parameters with the highest validation score:
```php
public best() : array
```

Return the underlying estimator trained with the best parameters:
```php
public estimator() : Estimator
```

##### Example:
```php
use Rubix\ML\GridSearch;
use Rubix\ML\Classifiers\KNearestNeighbors;
use Rubix\ML\Kernels\Distance\Euclidean;
use Rubix\ML\Kernels\Distance\Manhattan;
use Rubix\ML\CrossValidation\Metrics\Accuracy;
use Rubix\ML\CrossValidation\KFold;

...
$params = [
	[1, 3, 5, 10], [new Euclidean(), new Manhattan()],
];

$estimator = new GridSearch(KNearestNeightbors::class, $params, new Accuracy(), new KFold(10));

$estimator->train($dataset); // Train one estimator per parameter combination

var_dump($estimator->best()); // Return the best combination
```

##### Output:
```sh
array(2) {
  [0]=> int(5)
  [1]=> object(Rubix\ML\Kernels\Distance\Euclidean)#15 (0) {
  }
}
```

### Random Search
Random search is a hyperparameter selection technique that samples *n* parameters randomly from a user-specified distribution. In Rubix, the Random Params helper can be used along with [Grid Search](#grid-search) to achieve the goal of random search. The Random Params helper automatically takes care of deduplication so you never need to worry about testing a parameter twice. For this reason, however, you cannot generate more parameters than in range of, thus generating 5 unique ints between 1 and 3 is impossible.

To generate a distribution of integer parameters:
```php
public static ints(int $min, int $max, int $n = 10) : array
```

To generate a distribution of floating point parameters:
```php
public static floats(float $min, float $max, int $n = 10) : array
```

##### Example:
```php
use Rubix\ML\GridSearch;
use Rubix\ML\Other\Helpers\RandomParams;
use Rubix\ML\Clusterers\FuzzyCMeans;
use Rubix\ML\Kernels\Distance\Diagonal;
use Rubix\ML\Kernels\Distance\Minkowski;
use Rubix\CrossValidation\KFold;
use Rubix\CrossValidation\Metrics\VMeasure;

...
$params = [
	[1, 2, 3, 4, 5], RandomParams::floats(1.0, 20.0, 20), [new Diagonal(), new Minkowski(3.0)],
];

$estimator = new GridSearch(FuzzyCMeans::class, $params, new VMeasure(), new KFold(10));

$estimator->train($dataset);

var_dump($estimator->best());
```

##### Output:

```sh
array(3) {
  [0]=> int(4)
  [1]=> float(13.65)
  [2]=> object(Rubix\ML\Kernels\Distance\Diagonal)#15 (0) {
  }
}
```

---
### Model Persistence
Model persistence is the practice of saving a trained model to disk so that it can be restored later, on a different machine, or used in an online system. Rubix persists your models using built in PHP object serialization (similar to pickling in Python). Most Estimators are persistable, but some are not allowed due to their poor storage complexity.

### Persistent Model
It is possible to persist a model to disk by wrapping the Estimator instance in a Persistent Model meta-Estimator. The Persistent Model class gives the Estimator two additional methods `save()` and `restore()` that serialize and unserialize to and from disk. In order to be persisted the Estimator must implement the Persistable interface.

```php
public save(string $path) : bool
```
Where path is the location of the directory where you want the model saved. `save()` will return true if the model was successfully persisted and false if it failed.

```php
public static restore(string $path) : self
```
The restore method will return an instantiated model from the save path.

##### Example:
```php
use Rubix\ML\PersistentModel;
use Rubix\ML\Classifiers\RandomForest;

$estimator = new PersistentModel(new RandomForest(100, 0.2, 10, 3));

$estimator->save('path/to/models/folder/random_forest.model');

$estimator->save(); // Saves to current working directory under unique filename

$estimator = PersistentModel::restore('path/to/models/folder/random_forest.model');
```

### Transformers
Transformers take a sample matrices and transform them in various ways. A common transformation is scaling and centering the values using one of the Standardizers ([Z Scale](#z-scale-standardizer), [Robust](#robust-standardizer), [Quartile](#quartile-standardizer)). Transformers can be used with the [Pipeline](#pipeline) meta-Estimator or they can be used separately.

The fit method will allow the transformer to compute any necessary information from the training set in order to carry out its transformations. You can think of *fitting* a Transformer like *training* an Estimator. Not all Transformers need to be fit to the training set, when in doubt, call `fit()` anyways.
```php
public fit(Dataset $dataset) : void
```

The Transformer directly modifies a sample matrix via the `transform()` method.

```php
public transform(array $samples) : void
```

To transform a Dataset without having to pass the raw sample matrix you can call `apply()` on any Dataset object and it will apply the transformation to the underlying sample matrix automatically.

##### Example:
```php
use Rubix\ML\Transformers\MinMaxNormalizer;

...
$transformer = new MinMaxNormalizer();

$dataset->apply($transformer);
```

Here are a list of the Transformers available in Rubix.

### Dense and Sparse Random Projectors
A Random Projector is a dimensionality reducer based on the [Johnson-Lindenstrauss lemma](https://en.wikipedia.org/wiki/Johnson-Lindenstrauss_lemma "Johnson-Lindenstrauss lemma") that uses a random matrix to project a feature vector onto a user-specified number of dimensions. It is faster than most non-randomized dimensionality reduction techniques and offers similar performance.

The difference between the Dense and Sparse Random Projectors are that the Dense version uses a dense random guassian distribution and the Sparse version uses a sparse matrix (mostly 0's).

##### Continuous *Only*

##### Parameters:
| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | dimensions | None | int | The number of target dimensions to project onto. |

##### Additional Methods:
This Transformer does not have any additional methods.

##### Example:
```php
use Rubix\ML\Transformers\DenseRandomProjector;
use Rubix\ML\Transformers\SparseRandomProjector;

$transformer = new DenseRandomProjector(50);

$transformer = new SparseRandomProjector(50);
```

### L1 and L2 Regularizers
Augment each sample vector in the sample matrix such that each feature is divided over the L1 or L2 norm (or *magnitude*) of that vector.

##### Continuous *Only*

##### Parameters:
This Transformer does not have any parameters.

##### Additional Methods:
This Transformer does not have any additional methods.

##### Example:
```php
use Rubix\ML\Transformers\L1Regularizer;
use Rubix\ML\Transformers\L2Regularizer;

$transformer = new L1Regularizer();
$transformer = new L2Regularizer();
```

### Lambda Function
Run a stateless lambda function (*anonymous* function) over the sample matrix. The lambda function receives the sample matrix as an argument and should return the transformed sample matrix.

##### Categorical or Continuous

##### Parameters:
| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | lambda | None | callable | The lambda function to run over the sample matrix. |

##### Additional Methods:
This Transformer does not have any additional methods.

##### Example:
```php
use Rubix\ML\Transformers\LambdaFunction;

// Instantiate a lambda function that will sum up all the features for each sample
$transformer = new LambdaFunction(function ($samples) {
	return array_map(function ($sample) {
		return [array_sum($sample)];
	}, $samples);
});
```

### Min Max Normalizer
Often used as an alternative to [Standard Scaling](#z-scale-standardizer), the Min Max Normalization scales the input features to a range of between 0 and 1 by dividing the feature value over the maximum value for that feature column.

##### Continuous

##### Parameters:
This Transformer does not have any parameters.

##### Additional Methods:
This Transformer does not have any additional methods.

##### Example:
```php
use Rubix\ML\Transformers\MinMaxNormalizer;

$transformer = new MinMaxNormalizer();
```

### Missing Data Imputer
In the real world, it is common to have data with missing values here and there. The Missing Data Imputer replaces missing value placeholders with a guess based on a given guessing [Strategy](#guessing-strategies).

##### Categorical or Continuous

##### Parameters:
| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | placeholder | '?' | string or numeric | The placeholder that denotes a missing value. |
| 2 | continuous strategy | BlurryMean | object | The guessing strategy to employ for continuous feature columns. |
| 3 | categorical strategy | PopularityContest | object | The guessing strategy to employ for categorical feature columns. |

##### Additional Methods:
This Transformer does not have any additional methods.

##### Example:
```php
use Rubix\ML\Transformers\MissingDataImputer;
use Rubix\ML\Transformers\Strategies\BlurryMean;
use Rubix\ML\Transformers\Strategies\PopularityContest;

$transformer = new MissingDataImputer('?', new BlurryMean(0.2), new PopularityContest());
```

### Numeric String Converter
This handy Transformer will convert all numeric strings into their floating point counterparts. Useful for when extracting from a source that only recognizes data as string types.

##### Categorical

##### Parameters:
This Transformer does not have any parameters.

##### Additional Methods:
This Transformer does not have any additional methods.

##### Example:
```php
use Rubix\ML\Transformers\NumericStringConverter;

$transformer = new NumericStringConverter();
```

### One Hot Encoder
The One Hot Encoder takes a column of categorical features and produces a one-hot vector of n-dimensions where n is equal to the number of unique categories per feature column. This is used when you need to convert all features to continuous format since some Estimators do not work with categorical features.

##### Categorical

##### Parameters:
| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | columns | Null | array | The user-specified columns to encode indicated by numeric index starting at 0. |

##### Additional Methods:
This Transformer does not have any additional methods.

##### Example:
```php
use Rubix\ML\Transformers\OneHotEncoder;

$transformer = new OneHotEncoder([0, 3, 5, 7, 9]);
```

### Polynomial Expander
This Transformer will generate polynomial features up to and including the specified degree. Polynomial expansion is often used to fit data that is non-linear using a linear Estimator such as [Ridge](#ridge).

##### Continuous *Only*

##### Parameters:
| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | degree | 2 | int | The highest degree polynomial to generate from each feature vector. |

##### Additional Methods:
This Transformer does not have any additional methods.

##### Example:
```php
use Rubix\ML\Transformers\PolynomialExpander;

$transformer = new PolynomialExpander(3);
```

### Quartile Standardizer

This standardizer removes the median and scales each sample according to the interquantile range (*IQR*). The IQR is the range between the 1st quartile (25th *quantile*) and the 3rd quartile (75th *quantile*).

##### Continuous

##### Parameters:
This Transformer does not have any parameters.

##### Additional Methods:

Return the medians calculated by fitting the training set:
```php
public medians() : array
```

Return the interquartile ranges calculated during fitting:
```php
public iqrs() : array
```

##### Example:
```php
use Rubix\ML\Transformers\QuartileStandardizer;

$transformer = new QuartileStandardizer();
```

### Robust Standardizer
This Transformer standardizes continuous features by removing the median and dividing over the median absolute deviation (MAD), a value referred to as robust z score. The use of robust statistics makes this standardizer more immune to outliers than the [Z Scale Standardizer](#z-scale-standardizer).

##### Continuous

##### Parameters:
This Transformer does not have any parameters.

##### Additional Methods:

Return the medians calculated by fitting the training set:
```php
public medians() : array
```

Return the median absolute deviations calculated during fitting:
```php
public mads() : array
```

##### Example:
```php
use Rubix\ML\Transformers\RobustStandardizer;

$transformer = new RobustStandardizer();
```

### TF-IDF Transformer
Term Frequency - Inverse Document Frequency is a measure of how important a word is to a document. The TF-IDF value increases proportionally with the number of times a word appears in a document and is offset by the frequency of the word in the corpus. This Transformer makes the assumption that the input is made up of word frequency vectors such as those created by the [Count Vectorizer](#count-vectorizer).

##### Continuous *Only*

##### Parameters:
This Transformer does not have any parameters.

##### Additional Methods:
This Transformer does not have any additional methods.

##### Example:
```php
$transformer = new TfIdfTransformer();
```

### Variance Threshold Filter
A type of feature selector that removes all columns that have a lower variance than the threshold. Variance is computed as the population variance of all the values in the feature column.

##### Categorical and Continuous

##### Parameters:
| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | threshold | 0.0 | float | The threshold at which lower scoring columns will be dropped from the dataset. |

##### Additional Methods:

Return the columns that were selected during fitting:
```php
public selected() : array
```

##### Example:
```php
use Rubix\ML\Transformers\VarianceThresholdFilter;

$transformer = new VarianceThresholdFilter(50);
```

### Z Scale Standardizer
A way of centering and scaling an input vector by computing the Z Score for each continuous feature.

##### Continuous

##### Parameters:
This Transformer does not have any parameters.

##### Additional Methods:

Return the means calculated by fitting the training set:
```php
public means() : array
```

Return the standard deviations calculated during fitting:
```php
public stddevs() : array
```

##### Example:
```php
use Rubix\ML\Transformers\ZScaleStandardizer;

$transformer = new ZScaleStandardizer();
```

---
### Neural Network
A number of the Estimators in Rubix are implemented as a computational graph commonly referred to as a Neural Network due to its inspiration from the human brain. Neural Nets are trained using an iterative process called Gradient Descent and use Backpropagation (sometimes called Reverse Mode Autodiff) to calculate the error of each parameter in the network.

The [Multi Layer Perceptron](#multi-layer-perceptron) and [MLP Regressor](#mlp-regressor) are both neural networks capable of being built with an almost limitless combination of [Hidden layers](#hidden) employing various Activation Functions. The strength of deep neural nets (with 1 or more hidden layers) is its diversity in handling large amounts of data. In general, the deeper the neural network, the better it will perform.

### Activation Functions
The input to every neuron is passed through an Activation Function which determines its output. There are different properties of Activation Functions that make them more or less desirable depending on your problem.

### ELU
Exponential Linear Units are a type of rectifier that soften the transition from non-activated to activated using the exponential function.

##### Parameters:
| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | alpha | 1.0 | float | The value at which leakage will begin to saturate. Ex. alpha = 1.0 means that the output will never be more than -1.0 when inactivated. |

##### Example:
```php
use Rubix\ML\NeuralNet\ActivationFunctions\ELU;

$activationFunction = new ELU(5.0);
```

### Hyperbolic Tangent
S-shaped function that squeezes the input value into an output space between -1 and 1 centered at 0.

##### Parameters:
This Activation Function does not have any parameters.

##### Example:
```php
use Rubix\ML\NeuralNet\ActivationFunctions\HyperbolicTangent;

$activationFunction = new HyperbolicTangent();
```

### Identity
The Identity function (sometimes called Linear Activation Function) simply outputs the value of the input.

##### Parameters:
This Activation Function does not have any parameters.

##### Example:
```php
use Rubix\ML\NeuralNet\ActivationFunctions\Identity;

$activationFunction = new Identity();
```

### ISRU
Inverse Square Root units have a curve similar to [Hyperbolic Tangent](#hyperbolic-tangent) and [Sigmoid](#sigmoid) but use the inverse of the square root function instead. It is purported by the authors to be computationally less complex than either of the aforementioned. In addition, ISRU allows the parameter alpha to control the range of activation such that it equals + or - 1 / sqrt(alpha).

##### Parameters:
| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | alpha | 1.0 | float | The parameter that controls the range of activation. |

##### Example:
```php
use Rubix\ML\NeuralNet\ActivationFunctions\ISRU;

$activationFunction = new ISRU(2.0);
```

### Leaky ReLU
Leaky Rectified Linear Units are functions that output x when x > 0 or a small leakage value when x < 0. The amount of leakage is controlled by the user-specified parameter.

##### Parameters:
| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | leakage | 0.01 | float | The amount of leakage as a ratio of the input value. |

##### Example:
```php
use Rubix\ML\NeuralNet\ActivationFunctions\LeakyReLU;

$activationFunction = new LeakyReLU(0.001);
```

### SELU
Scaled Exponential Linear Unit is a self-normalizing activation function based on [ELU](#elu).

##### Parameters:
| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | scale | 1.05070 | float | The factor to scale the output by. |
| 2 | alpha | 1.67326 | float | The value at which leakage will begin to saturate. Ex. alpha = 1.0 means that the output will never be more than -1.0 when inactivated. |

##### Example:
```php
use Rubix\ML\NeuralNet\ActivationFunctions\SELU;

$activationFunction = new SELU(1.05070, 1.67326);
```

### Sigmoid
A bounded S-shaped function (specifically the Logistic function) with an output value between 0 and 1.

##### Parameters:
This Activation Function does not have any parameters.

##### Example:
```php
use Rubix\ML\NeuralNet\ActivationFunctions\Sigmoid;

$activationFunction = new Sigmoid();
```

### Soft Plus
A smooth approximation of the ReLU function whose output is constrained to be positive.

##### Parameters:
This Activation Function does not have any parameters.

##### Example:
```php
use Rubix\ML\NeuralNet\ActivationFunctions\SoftPlus;

$activationFunction = new SoftPlus();
```

### Softsign
A function that squashes the output of a neuron to + or - 1 from 0. In other words, the output is between -1 and 1.

##### Parameters:
This Activation Function does not have any parameters.

##### Example:
```php
use Rubix\ML\NeuralNet\ActivationFunctions\Softsign;

$activationFunction = new Softsign();
```

---
### Layers
Every network is made up of layers of computational units called neurons. Each layer processes and transforms the input from the previous layer.

There are three types of Layers that form a network, **Input**, **Hidden**, and **Output**. A network can have as many Hidden layers as the user specifies, however, there can only be 1 Input and 1 Output layer per network.

##### Example:
```php
use Rubix\ML\NeuralNet\Network;
use Rubix\ML\NeuralNet\Layers\Input;
use Rubix\ML\NeuralNet\Layers\Dense;
use Rubix\ML\NeuralNet\Layers\Softmax;
use Rubix\ML\NeuralNet\ActivationFunctions\ELU;
use Rubix\ML\NeuralNet\Optimizers\Adam;

$network = new Network(new Input(784), [
	new Dense(100, new ELU()),
	new Dense(100, new ELU()),
	new Dense(100, new ELU()),
], new Softmax([
	'dog', 'cat', 'frog', 'car',
], 1e-4), new Adam(0.001));
```

### Input
The Input Layer is simply a placeholder layer that represents the value of a sample or batch of samples. The number of placeholder nodes should be equal to the number of feature columns of a sample.

### Hidden
In multilayer networks, Hidden layers perform the bulk of the computation. They are responsible for transforming the input space in such a way that can be linearly separable by the Output layer. The more complex the problem space is, the more Hidden layers and neurons will be necessary to handle the complexity.

### Dense
Dense layers are fully connected Hidden layers, meaning each neuron is connected to each other neuron in the previous layer. Dense layers are able to employ a variety of [Activation Functions](#activation-functions) that modify the output of each neuron in the layer.

##### Parameters:
| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | neurons | None | int | The number of neurons in the layer. |
| 2 | activation fn | None | object | The activation function to use. |

##### Example:
```php
use Rubix\ML\NeuralNet\Layers\Dense;
use Rubix\ML\NeuralNet\ActivationFunctions\LeakyReLU;

$layer = new Dense(100, new LeakyReLU(0.05));
```

### Output
Activations are read directly from the Output layer when it comes to making a prediction. The type of Output layer used will determine the type of Estimator the Neural Net can power (Binary Classifier, Multiclass Classifier, or Regressor). The different types of Output layers are listed below.

### Linear
The Linear Output Layer consists of a single linear neuron that outputs a continuous scalar value useful for Regression problems.

##### Parameters:
| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | alpha | 1e-4 | float | The L2 regularization penalty. |

##### Example:
```php
use Rubix\ML\NeuralNet\Layers\Linear;

$layer = new Linear(1e-5);
```

### Logit
This Logit layer consists of a single [Sigmoid](#sigmoid) neuron capable of distinguishing between two classes. The Logit layer is useful for neural networks that output a binary class prediction.

##### Parameters:
| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | classes | None | array | The unique class labels of the binary classification problem. |
| 2 | alpha | 1e-4 | float | The L2 regularization penalty. |

##### Example:
```php
use Rubix\ML\NeuralNet\Layers\Logit;

$layer = new Logit(['yes', 'no'], 1e-5);
```

### Softmax
A generalization of the Logistic Layer, the Softmax Output Layer gives a joint probability estimate of a multiclass classification problem.

##### Parameters:
| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | classes | None | array | The unique class labels of the multiclass classification problem. |
| 2| alpha | 1e-4 | float | The L2 regularization penalty. |

##### Example:
```php
use Rubix\ML\NeuralNet\Layers\Softmax;

$layer = new Softmax(['yes', 'no', 'maybe'], 1e-6);
```

---
### Optimizers
Gradient Descent is an algorithm that takes iterative steps towards minimizing an objective function. There have been many papers that describe enhancements to the standard Stochastic Gradient Descent algorithm whose methods are encapsulated in pluggable Optimizers by Rubix. More specifically, Optimizers control the amount of Gradient Descent step to take for each parameter in the network upon each training iteration.

### AdaGrad
Short for *Adaptive Gradient*, the AdaGrad Optimizer speeds up the learning of parameters that do not change often and slows down the learning of parameters that do enjoy heavy activity.

##### Parameters:
| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | rate | 0.001 | float | The learning rate. i.e. the master step size. |

##### Example:
```php
use Rubix\ML\NeuralNet\Optimizers\AdaGrad;

$optimizer = new AdaGrad(0.035);
```

### Adam
Short for *Adaptive Momentum Estimation*, the Adam Optimizer uses both Momentum and RMS properties to achieve a balance of velocity and stability.

##### Parameters:
| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | rate | 0.001 | float | The learning rate. i.e. the master step size. |
| 2 | momentum | 0.9 | float | The decay rate of the Momentum property. |
| 3 | rms | 0.999 | float | The decay rate of the RMS property. |
| 4 | epsilon | 1e-8 | float | The smoothing constant used for numerical stability. |

##### Example:
```php
use Rubix\ML\NeuralNet\Optimizers\Adam;

$optimizer = new Adam(0.0001, 0.9, 0.999, 1e-8);
```

### Momentum
Momentum adds velocity to each step until exhausted. It does so by accumulating momentum from past updates and adding a factor to the current step.

##### Parameters:
| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | rate | 0.001 | float | The learning rate. i.e. the master step size. |
| 2 | decay | 0.9 | float | The Momentum decay rate. |

##### Example:
```php
use Rubix\ML\NeuralNet\Optimizers\Momentum;

$optimizer = new Momentum(0.001, 0.925);
```

### RMS Prop
An adaptive gradient technique that divides the current gradient over a rolling window of magnitudes of recent gradients.

##### Parameters:
| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | rate | 0.001 | float | The learning rate. i.e. the master step size. |
| 2 | decay | 0.9 | float | The RMS decay rate. |

##### Example:
```php
use Rubix\ML\NeuralNet\Optimizers\RMSProp;

$optimizer = new RMSProp(0.01, 0.9);
```

### Step Decay
A learning rate decay stochastic optimizer that reduces the learning rate by a factor of the decay parameter when it reaches a new floor (takes k steps).

##### Parameters:
| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | rate | 0.001 | float | The learning rate. i.e. the master step size. |
| 2 | k | 10 | int | The size of every floor in steps. i.e. the number of steps to take before applying another factor of decay. |
| 3 | decay | 1e-5 | float | The decay factor to decrease the learning rate by every k steps. |

##### Example:
```php
use Rubix\ML\NeuralNet\Optimizers\StepDecay;

$optimizer = new StepDecay(0.001, 15, 1e-5);
```

### Stochastic
A constant learning rate Optimizer.

##### Parameters:
| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | rate | 0.001 | float | The learning rate. i.e. the master step size. |

##### Example:
```php
use Rubix\ML\NeuralNet\Optimizers\Stochastic;

$optimizer = new Stochastic(0.001);
```

### Snapshots
Snapshots are a way to capture the state of a neural network at a moment in time. A Snapshot object holds all of the parameters in the network and can be used to restore the network back to a previous state.

To take a snapshot of your network simply call the `read()` method on the Network object. To restore the network from a snapshot pass the snapshot to the `restore()` method.

The example below shows how to take a snapshot and then restore the network via the snapshot.
```php
...
$snapshot = $network->read();

...

$network->restore($snapshot);
...
```

---
### Kernel Functions
Kernel functions are used to compute the similarity or distance between two vectors and can be plugged in to a particular Estimator to perform a part of the computation. They are pairwise positive semi-definite functions meaning their output is always 0 or greater. When considered as a hyperparameter, different Kernel functions have properties that can lead to different training and predictions.

### Distance
Distance functions are a type of Kernel that measures the distance between two coordinate vectors. They can be used throughout Rubix in Estimators that use the concept of distance to make predictions such as [K Nearest Neighbors](#k-nearest-neighbors), [K Means](#k-means), and [Local Outlier Factor](#local-outlier-factor).

### Canberra
A weighted version of [Manhattan](#manhattan) distance which computes the L1 distance between two coordinates in a vector space.

##### Parameters:
This Kernel does not have any parameters.

##### Example:
```php
use Rubix\ML\Kernels\Distance\Canberra;

$kernel = new Canberra();
```

### Cosine
Cosine Similarity is a measure that ignores the magnitude of the distance between two vectors thus acting as strictly a judgement of orientation. Two vectors with the same orientation have a cosine similarity of 1, two vectors oriented at 90° relative to each other have a similarity of 0, and two vectors diametrically opposed have a similarity of -1. To be used as a distance function, we subtract the Cosine Similarity from 1 in order to satisfy the positive semi-definite condition, therefore the Cosine *distance* is a number between 0 and 2.

##### Parameters:
This Kernel does not have any parameters.

##### Example:
```php
use Rubix\ML\Kernels\Distance\Cosine;

$kernel = new Cosine();
```

### Diagonal
The Diagonal (sometimes called Chebyshev) distance is a measure that constrains movement to horizontal, vertical, and diagonal from a point. An example that uses Diagonal movement is a chess board.

##### Parameters:
This Kernel does not have any parameters.

##### Example:
```php
use Rubix\ML\Kernels\Distance\Diagonal;

$kernel = new Diagonal();
```

### Ellipsoidal
The Ellipsoidal distance measures the distance between two points on a 3-dimensional ellipsoid.

##### Parameters:
This Kernel does not have any parameters.

##### Example:
```php
use Rubix\ML\Kernels\Distance\Ellipsoidal;

$kernel = new Ellipsoidal();
```

### Euclidean
This is the ordinary straight line (*bee line*) distance between two points in Euclidean space. The associated norm of the Euclidean distance is called the L2 norm.

##### Parameters:
This Kernel does not have any parameters.

##### Example:
```php
use Rubix\ML\Kernels\Distance\Euclidean;

$kernel = new Euclidean();
```

### Hamming
The Hamming distance is defined as the sum of all coordinates that are not exactly the same. Therefore, two coordinate vectors a and b would have a Hamming distance of 2 if only one of the three coordinates were equal between the vectors.


##### Parameters:
This Kernel does not have any parameters.

##### Example:
```php
use Rubix\ML\Kernels\Distance\Hamming;

$kernel = new Hamming();
```


### Manhattan
A distance metric that constrains movement to horizontal and vertical, similar to navigating the city blocks of Manhattan. An example that used this type of movement is a checkers board.

##### Parameters:
This Kernel does not have any parameters.

##### Example:
```php
use Rubix\ML\Kernels\Distance\Manhattan;

$kernel = new Manhattan();
```

### Minkowski
The Minkowski distance is a metric in a normed vector space which can be considered as a generalization of both the [Euclidean](#euclidean) and [Manhattan](#manhattan) distances. When the l*ambda* parameter is set to 1 or 2, the distance is equivalent to Manhattan and Euclidean respectively.

##### Parameters:
| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | lambda | 3.0 | float | Controls the curvature of the unit circle drawn from a point at a fixed distance. |

##### Example:
```php
use Rubix\ML\Kernels\Distance\Minkowski;

$kernel = new Minkowski(4.0);
```

---
### Cross Validation
Cross validation is the process of testing the generalization performance of a computer model using various techniques. Rubix has a number of classes that run cross validation on an instantiated Estimator for you. Each Validator outputs a scalar score based on the chosen metric.

### Validators
Validators take an [Estimator](#estimators) instance, [Labeled Dataset](#labeled) object, and validation [Metric](#validation-metrics) and return a validation score that measures the generalization performance of the model using one of various cross validation techniques. There is no need to train the Estimator beforehand as the Validator will automatically train it on subsets of the dataset chosen by the testing algorithm.

```php
public test(Estimator $estimator, Labeled $dataset, Validation $metric) : float
```

##### Example:
```php
use Rubix\ML\CrossValidation\KFold;
use Rubix\ML\CrossValidation\Metrics\Accuracy;

...
$validator = new KFold(10);

$score = $validator->test($estimator, $dataset, new Accuracy());

var_dump($score);
```

##### Output:
```sh
float(0.869)
```
Below describes the various Cross Validators available in Rubix.

### Hold Out
Hold Out is the simplest form of cross validation available in Rubix. It uses a *hold out* set equal to the size of the given ratio of the entire training set to test the model. The advantages of Hold Out is that it is quick, but it doesn't allow the model to train on the entire training set.

##### Parameters:
| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | ratio | 0.2 | float | The ratio of samples to hold out for testing. |

##### Example:
```php
use Rubix\ML\CrossValidation\HoldOut;
use Rubix\ML\CrossValidation\Metrics\Accuracy;

$validator = new HoldOut(0.25);
```

### K Fold
K Fold is a technique that splits the training set into K individual sets and for each training round uses 1 of the folds to measure the validation performance of the model. The score is then averaged over K. For example, a K value of 10 will train and test 10 versions of the model using a different testing set each time.

##### Parameters:
| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | k | 10 | int | The number of times to split the training set into equal sized folds. |

##### Example:
```php
use Rubix\ML\CrossValidation\KFold;

$validator = new KFold(5);
```

### Leave P Out
Leave P Out tests the model with a unique holdout set of P samples for each round until all samples have been tested. Note that this process can become slow with large datasets and small values of P.

##### Parameters:
| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | p | 10 | int | The number of samples to leave out each round for testing. |

##### Example:
```php
use Rubix\ML\CrossValidation\LeavePOut;

$validator = new LeavePOut(30);
```

### Validation Metrics

Validation metrics are for evaluating the performance of an Estimator given some ground truth such as class labels. The output of the Metric's `score()` method is a scalar score. You can output a tuple of minimum and maximum scores with the `range()` method.

To compute a validation score on an Estimator with a Labeled Dataset:
```php
public score(Estimator $estimator, Labeled $testing) : float
```

To output the range of values the metric can take on:
```php
public range() : array
```

##### Example:
```php
use Rubix\ML\CrossValidation\Metrics\MeanAbsoluteError;

...
$metric = new MeanAbsoluteError();

$score = $metric->score($estimator, $testing);

var_dump($metric->range());

var_dump($score);
```

##### Output:
```sh
array(2) {
  [0]=> float(-INF)
  [1]=> int(0)
}

float(-0.99846070553066)
```

There are different metrics for the different types of Estimators listed below.

### Anomaly Detection
| Metric | Range |  Description |
|--|--|--|
| Accuracy | (0, 1) | A quick metric that computes the accuracy of the detector. |
| F1 Score | (0, 1) | A metric that takes the precision and recall  into consideration. |

### Classification
| Metric | Range |  Description |
|--|--|--|
| Accuracy | (0, 1) | A quick metric that computes the accuracy of the classifier. |
| F1 Score | (0, 1) | A metric that takes the precision and recall of into consideration. |
| Informedness | (0, 1) | Measures the probability of making an informed prediction by looking at the sensitivity and specificity of each class outcome. |
| MCC | (-1, 1) | Matthews Correlation Coefficient is a coefficient between the observed and predicted binary classifications. A coefficient of +1 represents a perfect prediction, 0 no better than random prediction, and −1 indicates total disagreement between prediction and label. |


### Clustering
| Metric | Range | Description |
|--|--|--|
| Completeness | (0, 1) | A measure of the class outcomes that are predicted to be in the same cluster. |
| Concentration | (-INF, INF) | A score that measures the ratio between the within-cluster dispersion and the between-cluster dispersion (also called *Calinski Harabaz* score).
| Homogeneity | (0, 1) | A measure of the cluster assignments that are known to be in the same class. |
| V Measure | (0, 1) | The harmonic mean between Homogeneity and Completeness. |

##### Example:
```php
use Rubix\ML\CrossValidation\Metrics\Accuracy;
use Rubix\ML\CrossValidation\Metrics\Homogeneity;

$metric = new Accuracy();

$metric = new Homogeneity();
```

### Regression
| Metric | Range | Description |
|--|--|--|
| Mean Absolute Error | (-INF, 0) | The average absolute difference between the actual and predicted values. |
| Median Absolute Error | (-INF, 0) | The median absolute difference between the actual and predicted values. |
| Mean Squared Error | (-INF, 0) | The average magnitude or squared difference between the actual and predicted values. |
| RMS Error | (-INF, 0) | The root mean squared difference between the actual and predicted values. |
| R-Squared | (-INF, 1) | The R-Squared value, or sometimes called coefficient of determination is the proportion of the variance in the dependent variable that is predictable from the independent variable(s). |

---
### Reports
Reports allow you to evaluate the performance of a model with a testing set. To generate a report, pass a *trained* Estimator and a testing Dataset to the Report's `generate()` method. The output is an associative array that can be used to generate visualizations or other useful statistics.

To generate a report:
```php
public generate(Estimator $estimator, Dataset $dataset) : array
```

##### Example:
```php
use Rubix\ML\Classifiers\KNearestNeighbors;
use Rubix\ML\Reports\ConfusionMatrix;
use Rubix\ML\Datasets\Labeled;

...
$dataset = new Labeled($samples, $labels);

list($training, $testing) = $dataset->randomize()->split(0.8);

$estimator = new KNearestNeighbors(7);

$report = new ConfusionMatrix(['positive', 'negative', 'neutral']);

$estimator->train($training);

$result = $report->generate($estimator, $testing);
```

Most Reports require a Labeled dataset because they need some sort of ground truth to go compare to. The Reports that are available in Rubix are listed below.

### Aggregate Report
A Report that aggregates the results of multiple reports. The reports are indexed by their order given at construction time.

##### Parameters:
| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | reports | [ ] | array | An array of Report objects to aggregate. |

##### Example:
```php
use Rubix\ML\Reports\AggregateReport;
use Rubix\ML\Reports\ConfusionMatrix;
use Rubix\ML\Reports\MulticlassBreakdown;
use Rubix\ML\Reports\PredictionSpeed;

...
$report = new AggregateReport([
	new ConfusionMatrix(['wolf', 'lamb']),
	new MulticlassBreakdown(),
	new PredictionSpeed(),
]);

$result = $report->generate($estimator, $testing);
```

### Confusion Matrix
A Confusion Matrix is a table that visualizes the true positives, false, positives, true negatives, and false negatives of a Classifier. The name stems from the fact that the matrix makes it easy to see the classes that the Classifier might be confusing.

##### Classification

##### Parameters:
| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | classes | All | array | The classes to compare in the matrix. |

##### Example:
```php
use Rubix\ML\Reports\ConfusionMatrix;

...
$report = new ConfusionMatrix(['dog', 'cat', 'turtle']);

$result = $report->generate($estimator, $testing);

var_dump($result);
```

##### Output:

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

A Contingency Table is used to display the frequency distribution of class labels among a clustering of samples.

##### Clustering

##### Parameters:
This Report does not have any parameters.

##### Example:
```php
use Rubix\ML\Reports\ContingencyTable;

...
$report = new ContingencyTable();

$result = $report->generate($estimator, $testing);

var_dump($result);
```

##### Output:
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
A Report that drills down in to each unique class outcome. The report includes metrics such as Accuracy, F1 Score, MCC, Precision, Recall, Cardinality, Miss Rate, and more.

##### Classification

##### Parameters:
This Report does not have any parameters.

##### Example:
```php
use Rubix\ML\Reports\MulticlassBreakdown;

...
$report = new MulticlassBreakdown();

$result = $report->generate($estimator, $testing);

var_dump($result);
```

##### Output:
```sh
...
    array(2) {
      ["benign"]=>
      array(15) {
        ["accuracy"]=> int(1)
        ["precision"]=> float(0.99999999998723)
        ["recall"]=> float(0.99999999998723)
        ["specificity"]=> float(0.99999999998812)
        ["miss_rate"]=> float(1.2771450563775E-11)
        ["fall_out"]=> float(1.1876499783625E-11)
        ["f1_score"]=> float(0.99999999498723)
        ["mcc"]=> float(0.99999999999998)
        ["informedness"]=> float(0.99999999997535)
        ["true_positives"]=> int(783)
        ["true_negatives"]=> int(842)
        ["false_positives"]=> int(0)
        ["false_negatives"]=> int(0)
        ["cardinality"]=> int(783)
        ["density"]=> float(0.48184615384615)
      }
...
```

### Outlier Ratio
Outlier Ratio is the proportion of outliers to inliers in an [Anomaly Detection](#anomaly-detectors) problem. It can be used to gauge the amount of contamination that the Detector is predicting.

##### Anomaly Detection

##### Parameters:
This Report does not have any parameters.

##### Example:
```php
use Rubix\ML\Reports\OutlierRatio;

...
$report = new OutlierRatio();

$result = $report->generate($estimator, $testing);

var_dump($result);
```

##### Output:
```sh
  array(4) {
    ["outliers"]=> int(13)
    ["inliers"]=> int(307)
    ["ratio"]=> float(0.042345276871585)
    ["cardinality"]=> int(320)
  }
```

### Prediction Speed
This Report measures the number of predictions an Estimator can make per second as given by the PPS (predictions per second) score.

##### Classification, Regression, Clustering, Anomaly Detection

##### Parameters:
This Report does not have any parameters.

##### Example:
```php
use Rubix\ML\Reports\PredictionSpeed;

...
$report = new PredictionSpeed();

$result = $report->generate($estimator, $testing);

var_dump($result);
```

##### Output:
```sh
  array(4) {
    ["ppm"]=> float(4332968.1101788)
    ["average_seconds"]=> float(1.3847287706694E-5)
    ["total_seconds"]=> float(0.0041680335998535)
    ["cardinality"]=> int(301)
  }

```

### Residual Analysis
Residual Analysis is a type of Report that measures the differences between the predicted and actual values of a regression problem.

##### Regression

##### Parameters:
This Report does not have any parameters.

##### Example:
```php
use Rubix\ML\Reports\ResidualAnalysis;

...
$report = new ResidualAnalysis();

$result = $report->generate($estimator, $testing);

var_dump($result);
```

##### Output:
```sh
  array(9) {
    ["mean_absolute_error"]=>
    float(2.1971189157834)
    ["median_absolute_error"]=>
    float(1.714)
    ["mean_squared_error"]=>
    float(8.7020753279997)
    ["rms_error"]=>
    float(2.9499280208167)
    ["min"]=>
    float(0.0069999999999908)
    ["max"]=>
    float(14.943333333333)
    ["variance"]=>
    float(3.8747437979066)
    ["r_squared"]=>
    float(0.82286934000174)
    ["cardinality"]=>
    int(301)
  }
```

---
### Other
This section includes broader functioning objects that aren't part of a specific category.

### Guessing Strategies
Guesses can be thought of as a type of *weak* prediction. Unlike a real prediction, guesses are made using limited information and basic means. A guessing Strategy attempts to use such information to formulate an educated guess. Guessing is utilized in both Dummy Estimators ([Dummy Classifier](#dummy-classifier), [Dummy Regressor](#dummy-regressor)) as well as the [Missing Data Imputer](#missing-data-imputer).

The Strategy interface provides an API similar to Transformers as far as fitting, however, instead of being fit to an entire dataset, each Strategy is fit to an array of either continuous or discrete values.

To fit a Strategy to an array of values:
```php
public fit(array $values) : void
```

To make a guess based on the fitted data:
```php
public guess() : mixed
```

##### Example:
```php
use Rubix\ML\Other\Strategies\BlurryMedian;

$values = [1, 2, 3, 4, 5];

$strategy = new BlurryMedian(0.05);

$strategy->fit($values);

var_dump($strategy->range()); // Min and max guess for continuous strategies

var_dump($strategy->guess());
var_dump($strategy->guess());
var_dump($strategy->guess());
```

##### Output:
```sh
array(2) {
  [0]=> float(2.85)
  [1]=> float(3.15)
}

float(2.897176548)
float(3.115719462)
float(3.105983314)
```
Strategies are broken up into the Categorical type and the Continuous type. You can output the set of all possible categorical guesses by calling the `set()` method on any Categorical Strategy. Likewise, you can call `range()` on any Continuous Strategy to output the minimum and maximum values the guess can take on.

Here are the guessing Strategies available to use in Rubix.

### Blurry Mean
This continuous Strategy that adds a blur factor to the mean of a set of values producing a random guess around the mean.

##### Continuous

##### Parameters:
| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | blur | 0.2 | float | The amount of Gaussian noise by ratio of the standard deviation to add to the guess. |

##### Example:
```php
use Rubix\ML\Other\Strategies\BlurryMean;

$strategy = new BlurryMean(0.05);
```

### Blurry Median
Adds random Gaussian noise to the median of a set of values.

##### Continuous

##### Parameters:
| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | blur | 0.2 | float | The amount of Gaussian noise by ratio of the interquartile range to add to the guess. |

##### Example:
```php
use Rubix\ML\Other\Strategies\BlurryMedian;

$strategy = new BlurryMedian(0.5);
```

### K Most Frequent
This Strategy outputs one of K most frequent discrete values at random.

##### Categorical

##### Parameters:
| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | k | 1 | int | The number of most frequency categories to consider when formulating a guess. |

##### Example:
```php
use Rubix\ML\Other\Strategies\KMostFrequent;

$strategy = new KMostFrequent(5);
```

### Lottery
Hold a lottery in which each category has an equal chance of being picked.

##### Categorical

##### Parameters:
This Strategy does not have any parameters.

##### Example:
```php
use Rubix\ML\Other\Strategies\Lottery;

$strategy = new Lottery();
```

### Popularity Contest
Hold a popularity contest where the probability of winning (being guessed) is based on the category's prior probability.

##### Categorical

##### Parameters:
This Strategy does not have any parameters.

##### Example:
```php
use Rubix\ML\Other\Strategies\Lottery;

$strategy = new PopularityContest();
```

### Wild Guess
It's just what you think it is. Make a guess somewhere in between the minimum and maximum values observed during fitting.

##### Continuous

##### Parameters:
This Strategy does not have any parameters.

##### Example:
```php
use Rubix\ML\Other\Strategies\WildGuess;

$strategy = new WildGuess();
```

---
### Tokenizers
Tokenizers take a body of text and converts it to an array of string tokens. Tokenizers are used by various algorithms in Rubix such as the [Count Vectorizer](#count-vectorizer) to encode text into word counts.

To tokenize a body of text:
```php
public tokenize(string $text) : array
```

##### Example:
```php
use Rubix\ML\Other\Tokenizers\Word;

$text = 'I would like to die on Mars, just not on impact.';

$tokenizer = new Word();

var_dump($tokenizer->tokenize($text));
```

##### Output:
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

Below are the Tokenizers available in Rubix.

### Whitespace
Tokens are delimited by a user-specified whitespace character.

##### Parameters:
| # | Param | Default | Type | Description |
|--|--|--|--|--|
| 1 | delimiter | ' ' | string | The whitespace character that delimits each token. |

##### Example:
```php
use Rubix\ML\Other\Tokenizers\Whitespace;

$tokenizer = new Whitespace(',');
```

### Word Tokenizer
Tokens are matched via regular expression designed to pick out words from a block of text. Note that this tokenizer will only pick up on words that are 2 or more characters.

##### Parameters:
This Tokenizer does not have any parameters.

##### Example:
```php
use Rubix\ML\Other\Tokenizers\Word;

$tokenizer = new Word();
```

---
## Testing
Rubix utilizes a combination of static analysis and unit tests to reduce errors in code. Rubix provides two Composer scripts that can be run from the root directory that automate the testing process.

To run static analysis:
```sh
composer analyze
```

To run the unit tests:
```sh
composer test
```

---
## Contributing
Please make sure all your code is tested and passes static analysis (see [Testing](#testing) section above) before submitting it to the repository.

Rubix ML for PHP
PHP from Packagist Latest Stable Version Travis GitHub license

Rubix ML is a library that lets you build intelligent programs that learn from data in PHP.

Our Mission
The goal of Rubix is to bring easy to use machine learning (ML) capabilities to the PHP language. We aspire to provide the framework to facilitate small to medium sized projects, rapid prototyping, and education. If you would like to join in on the mission, you get up and running fast by following the instructions below.

Installation
Install Rubix using composer:

$ composer require rubix/ml
Requirements
PHP 7.1.3 or above
GD extension for Image Vectorization
License
MIT

Documentation
Table of Contents
Introduction
Obtaining Data
Choosing an Estimator
Training and Prediction
Evaluation
Next Steps
API Reference
Datasets
Dataset Objects
Labeled
Unlabeled
Generators
Agglomerate
Blob
Circle
Half Moon
Feature Extractors
Count Vectorizer
Pixel Encoder
Estimators
Anomaly Detectors
Isolation Forest
Isolation Tree
Local Outlier Factor
Robust Z Score
Classifiers
AdaBoost
Classification Tree
Dummy Classifier
Extra Tree
Gaussian Naive Bayes
K Nearest Neighbors
Logistic Regression
Multi Layer Perceptron
Naive Bayes
Random Forest
Softmax Classifier
Clusterers
DBSCAN
Fuzzy C Means
K Means
Mean Shift
Regressors
Dummy Regressor
KNN Regressor
MLP Regressor
Regression Tree
Ridge
Estimator Interfaces
Online
Probabilistic
Persistable
Meta-Estimators
Data Preprocessing
Pipeline
Ensemble
Bootstrap Aggregator
Committee Machine
Model Persistence
Persistent Model
Model Selection
Grid Search
Random Search
Transformers
Dense and Sparse Random Projectors
L1 and L2 Regularizers
Lambda Function
Min Max Normalizer
Missing Data Imputer
Numeric String Converter
One Hot Encoder
Polynomial Expander
Quartile Standardizer
Robust Standardizer
TF-IDF Transformer
Variance Threshold Filter
Z Scale Standardizer
Neural Network
Activation Functions
ELU
Hyperbolic Tangent
Identity
ISRU
Leaky ReLU
SELU
Sigmoid
Soft Plus
Softsign
Layers
Input
Hidden
Dense
Output
Linear
Logit
Softmax
Optimizers
AdaGrad
Adam
Momentum
RMS Prop
Step Decay
Stochastic
Snapshots
Kernel Functions
Distance
Canberra
Cosine
Diagonal
Ellipsoidal
Euclidean
Hamming
Manhattan
Minkowski
Cross Validation
Validators
Hold Out
K Fold
Leave P Out
Metrics
Anomaly Detection
Classification
Clustering
Regression
Reports
Aggregate Report
Confusion Matrix
Contingency Table
Multiclass Breakdown
Outlier Ratio
Prediction Speed
Residual Analysis
Other
Guessing Strategies
Blurry Mean
Blurry Median
K Most Frequent
Lottery
Popularity Contest
Wild Guess
Tokenizers
Whitespace
Word
An Introduction to Machine Learning in Rubix
Machine learning is the process by which a computer program is able to progressively improve performance on a certain task through training and data without explicitly being programmed. There are two types of learning that Rubix offers out of the box, Supervised and Unsupervised.

Supervised learning is a technique to train computer models with a dataset in which the outcome of each sample data point has been labeled either by a human expert or another ML model prior to training. There are two types of supervised learning to consider in Rubix:
Classification is the problem of identifying which class a particular sample belongs to. For example, one task may be in determining a particular species of Iris flower based on its sepal and petal dimensions.
Regression involves predicting continuous values rather than discrete classes. An example in which a regression model is appropriate would be predicting the life expectancy of a population based on economic factors.
Unsupervised learning, by contrast, uses an unlabeled dataset and relies on the information within the training samples to learn insights.
Clustering is the process of grouping data points in such a way that members of the same group are more similar (homogeneous) than the rest of the samples. You can think of clustering as assigning a class label to an otherwise unlabeled sample. An example where clustering might be used is in differentiating tissues in PET scan images.
Anomaly Detection is the flagging of samples that do not conform to an expected pattern. Anomalous samples can often indicate adversarial activity, bad data, or exceptional performance.
Obtaining Data
Machine learning projects typically begin with a question. For example, who of my friends are most likely to stay married to their spouse? One way to go about answering this question with machine learning would be to go out and ask a bunch of long-time married and divorced couples the same set of questions and then use that data to build a model of what a successful (or not) marriage looks like. Later, you can use that model to make predictions based on the answers from your friends.

Although this is certainly a valid way of obtaining data, in reality, chances are someone has already done the work of measuring the data for you and it is your job to find it, aggregate it, clean it, and otherwise make it usable by the machine learning algorithm. There are a number of PHP libraries out there that make extracting data from CSV, JSON, databases, and cloud services a whole lot easier, and we recommend checking them out before attempting it manually.

Having that said, Rubix will be able to handle any dataset as long as it can fit into one its predefined Dataset objects (Labeled, Unlabeled, etc.).

The Dataset Object
Data is passed around in Rubix via specialized data containers called Datasets. Dataset objects extend the PHP array structure with methods that properly handle selecting, splitting, folding, and randomizing the samples. In general, there are two types of Datasets, Labeled and Unlabeled. Labeled datasets are typically used for supervised learning and Unlabeled datasets are used for unsupervised learning and for making predictions.

For the following example, suppose that you went out and asked 100 couples (50 married and 50 divorced) to rate (between 1 and 5) their similarity, communication, and partner attractiveness. We could construct a Labeled Dataset object from the data you collected in the following way:

use \Rubix\ML\Datasets\Labeled;

$samples = [[3, 4, 2], [1, 5, 3], [4, 4, 3], [2, 1, 5], ...];

$labels = ['married', 'divorced', 'married', 'divorced', ...];

$dataset = new Labeled($samples, $labels);
The Dataset object is now ready to be used throughout Rubix.

Choosing an Estimator
There are many different algorithms to chose from in Rubix and each one is designed to handle specific (sometimes overlapping) tasks. Choosing the right Estimator for the job is crucial to building an accurate and performant computer model.

There are a couple ways that we could model our marriage satisfaction predictor. We could have asked a fourth question - that is, to rate each couple’s overall marriage satisfaction from say 1 to 10 and then train a Regressor to predict a continuous “satisfaction score” for each new sample. But since all we have to go by for now is whether they are still married or currently divorced, a Classifier will be better suited.

In practice, one will experiment with more than one type of Estimator to find the best fit to the data, but for the purposes of this introduction we will simply demonstrate a common and intuitive algorithm called K Nearest Neighbors.

Creating the Estimator Instance
Like most Estimators, the K Nearest Neighbors classifier requires a number of parameters (called Hyperparameters) to be chosen up front. These parameters can be selected based on some prior knowledge of the problem space, or at random. Rubix provides a meta-Estimator called Grid Search that searches the parameter space provided for the most effective combination. For the purposes of this example we will just go with our intuition and chose the parameters outright.

It is important to understand the effect that each parameter has on the performance of the Estimator since different settings can often lead to different results.

You can find a full description of all of the K Nearest Neighbors parameters in the API reference which we highly recommend reading a few times to get a grasp for what each parameter does.

The K Nearest Neighbors algorithm works by comparing the distance between a sample and each of the points from the training set. It will then use the K nearest points to base its prediction on. For example, if the 5 closest neighbors to a sample are 4 married and 1 divorced, the algorithm will output a prediction of married with a probability of 0.80.

To create a K Nearest Neighbors Classifier instance:

use Rubix\ML\Classifiers\KNearestNeighbors;
use Rubix\ML\Kernels\Distance\Manhattan;

// Using the default parameters
$estimator = new KNearestNeighbors();

// Specifying parameters
$estimator = new KNearestNeighbors(3, new Manhattan());
Now that we’ve chosen and instantiated an Estimator and our Dataset object is ready to go, it is time to train our model and use it to make some predictions.

Training and Prediction
Training is the process of feeding the Estimator data so that it can learn. The unique way in which the Estimator learns is based upon the underlying algorithm which has been implemented for you already. All you have to do is supply enough clean data so that the process can converge to a satisfactory optimum.

Passing the Labeled Dataset object we created earlier we can train the KNN estimator like so:

...
$estimator->train($dataset);
That’s it.

For our 100 sample dataset, this should only take a few microseconds, but larger datasets and more sophisticated Estimators can take much longer.

Once the Estimator has been fully trained we can feed in some new sample data points to see what the model predicts. Suppose that we went out and collected 5 new data points from our friends using the same questions we asked the couples we interviewed for our training set. We could make a prediction on whether they look more like the class of married or divorced couples by taking their answers and running them through the trained Estimator’s predict() method.

use Rubix\ML\Dataset\Unlabeled;

$samples = [[4, 1, 3], [2, 2, 1], [2, 4, 5], [5, 2, 4], [3, 2, 1];

$friends = new Unlabeled($samples);

$predictions = $estimator->predict($friends);

var_dump($predictions);
Output:
array(5) {
	[0] => 'divorced'
	[1] => 'divorced'
	[2] => 'married'
	[3] => 'married'
	[4] => 'divorced'
}
Note that we are not using a Labeled Dataset here because we don’t know the outcomes yet. In fact, the label is exactly what we are trying to predict. Next we’ll look at how we can test the accuracy of the predictions our model makes using cross validation.

Evaluating Model Performance
Making predictions is not very useful unless the Estimator can correctly generalize what it has learned during training. Cross Validation is a process by which we can test the model for its generalization ability. For the purposes of this introduction, we will use a simple form of cross validation called Hold Out. The Hold Out validator will take care of randomizing and splitting the dataset into training and testing sets automatically, such that a portion of the data is held out to be used to test (or validate) the model. The reason we do not use all of the data for training is because we want to test the Estimator on samples that it has never seen before.

Hold Out requires you to set the ratio of testing to training samples to use. In this case, let’s chose to use a factor of 0.2 (20%) of the dataset for testing leaving the rest (80%) for training. Typically, 0.2 is a good default choice however your mileage may vary. The important thing to understand here is the trade off between more data for training and more precise testing results. Once you get the hang of Hold Out, the next step is to consider more elaborate cross validation techniques such as K Fold, and Leave P Out.

To return a validation score from the Hold Out Validator using the Accuracy Metric just pass it the untrained Estimator instance and a dataset.

Example:
use Rubix\ML\CrossValidation\HoldOut;
use Rubix\ML\CrossValidation\Metrics\Accuracy;

...
$validator = new HoldOut(0.2);

$score = $validator->test($estimator, $dataset, new Accuracy());

var_dump($score);
Output:
float(0.945)
Since we are measuring accuracy, this output indicates that our Estimator is 94.5% accurate given the data we’ve trained and tested it with. Not bad.

What Next?
Now that you’ve gone through a brief introduction of a simple machine learning problem in Rubix, the next step is to become more familiar with the API and to experiment with some data on your own. We highly recommend reading the entire documentation at least twice to fully understand all of the features at your disposal. If you’re eager to get started, a great place to begin is by downloading some datasets from the University of California Irvine Machine Learning Repository where they have many pre-cleaned datasets available for free.

API Reference
Here you will find information regarding the classes that make up the Rubix library.

Datasets
Data is what powers machine learning programs so naturally we treat it as a first-class citizen. Rubix provides a number of classes that help you wrangle and even generate data.

Dataset Objects
In Rubix, data is passed around using specialized data structures called Dataset objects. Dataset objects can hold a heterogeneous mix of categorical and ncontinuous data and gracefully handles null values with a user-defined placeholder. Dataset objects make it easy to slice and transport data in a canonical way.

There are two types of data that Estimators can process i.e. categorical and continuous. Any numerical (integer or float) datum is considered continuous and any string datum is considered categorical as a general rule throughout Rubix. This rule makes it easy to distinguish between the types of data while allowing for flexibility. For example, you could represent the number 5 as continuous by using the integer type or as categorical by using the string type (‘5’).

Example:
use Rubix\ML\Datasets\Unlabeled;

$samples = [
	['rough', 8, 6.55], ['furry', 10, 9.89], ...
];

$dataset = new Unlabeled($samples);
Selecting
Return the sample matrix:

public samples() : array
Select the sample at row offset:

public row(int $index) : array
Select the values of a feature column at offset:

public column(int $index) : array
Return the first n rows of data in a new Dataset object:

public head(int $n = 10) : self
Return the last n rows of data in a new Dataset object:

public tail(int $n = 10) : self
Example:
// Return the sample matrix
$samples = $dataset->samples();

// Return just the first 5 rows in a new dataset
$subset = $dataset->head(5);
Properties
Return the number of rows in the Dataset:

public numRows() : int
Return the number of columns in the Dataset:

public numColumns() : int
Splitting, Folding, and Batching
Remove n rows from the Dataset and return them in a new Dataset:

public take(int $n = 1) : self
Leave n samples on the Dataset and return the rest in a new Dataset:

public leave(int $n = 1) : self
Split the Dataset into left and right subsets given by a ratio:

public split(float $ratio = 0.5) : array
Fold the Dataset k - 1 times to form k equal size Datasets:

public fold(int $k = 10) : array
Batch the Dataset into subsets of n rows per batch:

public batch(int $n = 50) : array
Example:
// Remove the first 5 rows and return them in a new dataset
$subset = $dataset->take(5);

// Split the dataset into left and right subsets
list($left, $right) = $dataset->split(0.5);

// Fold the dataset into 8 equal size datasets
$folds = $dataset->fold(8);
Randomizing
Randomize the order of the Dataset and return it:

public randomize() : self
Generate a random subset of size n:

public randomSubset($n = 1) : self
Generate a random subset with replacement of size n:

public randomSubsetWithReplacement($n = 1) : self
Example:
// Randomize and split the dataset into two subsets
list($left, $right) = $dataset->randomize()->split(0.8);

// Generate a dataset of 500 random samples
$subset = $dataset->randomSubset(500);
Sorting
To sort a Dataset by a specific feature column:

public sortByColumn(int $index, bool $descending = false) : self
Example:
...
var_dump($dataset->samples());

$dataset->sortByColumn(2, false);

var_dump($dataset->samples());
Output:
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
Prepending and Appending
To prepend a given Dataset onto the beginning of another Dataset:

public prepend(Dataset $dataset) : self
To append a given Dataset onto the end of another Dataset:

public append(Dataset $dataset) : self
Applying a Transformation
You can apply a fitted Transformer to a Dataset directly passing it to the apply method on the Dataset.

public apply(Transformer $transformer) : void
Example:
use Rubix\ML\Transformers\OneHotEncoder;

...
$transformer = new OneHotEncoder();

$transformer->fit($dataset);

$dataset->apply($transformer);
Saving and Restoring
Dataset objects can be saved and restored from a serialized object file which makes them easy to work with. Saving will capture the current state of the dataset including any transformations that have been applied.

Save the Dataset to a file:

public save(?string $path = null) : void
Restore the Dataset from a file:

public static restore(string $path) : self
Example:
// Save the dataset to a file
$dataset->save('path/to/dataset');

// Assign a filename (ex. 1531772454.dataset)
$dataset->save();

$dataset = Labeled::restore('path/to/dataset');
There are two types of Dataset objects in Rubix, labeled and unlabeled.

Labeled
For supervised Estimators you will need to pass it a Labeled Dataset consisting of a sample matrix and an array of labels that correspond to the observed outcomes of each sample. Splitting, folding, randomizing, sorting, and subsampling are all done while keeping the indices of samples and labels aligned.

In addition to the basic Dataset interface, the Labeled class can sort and stratify the data by label.

Parameters:
#	Param	Default	Type	Description
1	samples	None	array	A 2-dimensional array consisting of rows of samples and columns of features.
2	labels	None	array	A 1-dimensional array of labels that correspond to the samples in the dataset.
3	placeholder	‘?’	mixed	The placeholder value for null features.
Additional Methods:
Return a 1-dimensional array of labels:

public labels() : array
Return the label at the given row offset:

public label(int $index) : mixed
Return all of the possible outcomes i.e the unique labels:

public possibleOutcomes() : array
Sort the Dataset by label:

public sortByLabel(bool $descending = false) : self
Group the samples by label and return them in their own Dataset:

public stratify() : array
Split the Dataset into left and right stratified subsets with a given ratio of samples in each:

public stratifiedSplit($ratio = 0.5) : array
Fold the Dataset k - 1 times to form k equal size stratified Datasets

public stratifiedFold($k = 10) : array
Example:
use Rubix\ML\Datasets\Labeled;

...
$dataset = new Labeled($samples, $labels);

// Return all the labels in the dataset
$labels = $dataset->labels();

// Return the label at offset 3
$label = $dataset->label(3);

// Return all possible unique labels
$outcomes = $dataset->possibleOutcomes();

var_dump($labels);
var_dump($label);
var_dump($outcomes);
Output:
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
Example:
...
// Fold the dataset into 5 equal size stratified subsets
$folds = $dataset->stratifiedFold(5);

// Split the dataset into two stratified subsets
list($left, $right) = $dataset->stratifiedSplit(0.8);

// Put each sample with label x into its own dataset
$strata = $dataset->stratify();
Unlabeled
Unlabeled datasets can be used for training unsupervised Estimators and for feeding data into an Estimator to make predictions.

Parameters:
#	Param	Default	Type	Description
1	samples	None	array	A 2-dimensional feature matrix consisting of rows of samples and columns of feature values.
2	placeholder	‘?’	mixed	The placeholder value for null features.
Additional Methods:
This Dataset does not have any additional methods.

Example:
use Rubix\ML\Datasets\Unlabeled;

$dataset = new Unlabeled($samples);
Generators
Dataset Generators allow you to produce data of a user-specified shape, dimensionality, and cardinality. This is useful for augmenting a dataset with synthetic data or for testing and demonstration purposes.

To generate a Dataset object with n samples (rows):

public generate(int $n = 100) : Dataset
Return the dimensionality of the samples produced:

public dimensions() : int
Example:
use Rubix\ML\Datasets\Generators\Blob;

$generator = new Blob([0, 0], 1.0);

$dataset = $generator->generate(3);

var_dump($generator->dimensions());

var_dump($dataset->samples());
Output:
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
Agglomerate
An Agglomerate is a collection of other generators each given a label. Agglomerates are useful for classification, clustering, and anomaly detection problems where the label is a discrete value.

Parameters:
#	Param	Default	Type	Description
1	generators	[ ]	array	A collection of generators keyed by their user-specified label (0 indexed by default).
2	weights	1 / n	array	A set of arbitrary weight values corresponding to a generator’s contribution to the agglomeration.
Additional Methods:
Return the normalized weights of each generator in the agglomerate:

public weights() : array
Example:
use Rubix\ML\Datasets\Generators\Agglomerate;

$generator = new Agglomerate([
	new Blob([5, 2], 1.0),
	new HalfMoon([-3, 5], 1.5, 90.0, 0.1),
	new Circle([2, -4], 2.0, 0.05),
], [
	5, 6, 3, // Weights
]);
Blob
A normally distributed n-dimensional blob of samples centered at a given mean vector. The standard deviation can be set for the whole blob or for each feature column independently. When a global value is used, the resulting blob will be isotropic.

Parameters:
#	Param	Default	Type	Description
1	center	[ ]	array	The coordinates of the center of the blob i.e. a centroid vector.
2	stddev	1.0	float or array	Either the global standard deviation or an array with the SD for each feature column.
Additional Methods:
This Generator does not have any additional methods.

Example:
use Rubix\ML\Datasets\Generators\Blob;

$generator = new Blob([1.2, 5.0, 2.6, 0.8], 0.25);
Circle
Create a circle made of sample data points in 2 dimensions.

Parameters:
#	Param	Default	Type	Description
1	center	[ ]	array	The x and y coordinates of the center of the circle.
2	scale	1.0	float	The scaling factor of the circle.
3	noise	0.1	float	The amount of Gaussian noise to add to each data point as a ratio of the scaling factor.
Additional Methods:
This Generator does not have any additional methods.

Example:
use Rubix\ML\Datasets\Generators\Circle;

$generator = new Circle([0.0, 0.0], 100, 0.1);
Half Moon
Generate a dataset consisting of 2-d samples that form a half moon shape.

Parameters:
#	Param	Default	Type	Description
1	center	[ ]	array	The x and y coordinates of the center of the circle.
2	scale	1.0	float	The scaling factor of the circle.
3	rotate	90.0	float	The amount in degrees to rotate the half moon counterclockwise.
4	noise	0.1	float	The amount of Gaussian noise to add to each data point as a ratio of the scaling factor.
Additional Methods:
This Generator does not have any additional methods.

Example:
use Rubix\ML\Datasets\Generators\HalfMoon;

$generator = new HalfMoon([0.0, 0.0], 100, 180.0, 0.2);
Feature Extractors
Feature Extractors are objects that help you encode raw data into feature vectors so they can be used by an Estimator.

Extractors have an API similar to Transformers, however, they are designed to be used on the raw data before it is inserted into a Dataset Object. The output of the extract() method is a sample matrix that can be used to build a Dataset Object.

Fit the Extractor to the raw samples before extracting:

public fit(array $samples) : void
Return a sample matrix:

public extract(array $samples) : array
Example:
use Rubix\ML\Extractors\CountVectorizer;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Datasets\Labeled;

...
$estractor = new CountVectorizer(5000);

$extractor->fit($data);

$samples = $extractor->extract($data);

$dataset = new Unlabeled($samples);

$dataset = new Labeled($samples, $labels);
Count Vectorizer
In machine learning, word counts are often used to represent natural language as numerical vectors. The Count Vectorizer builds a vocabulary from the training samples during fitting and transforms an array of strings (text blobs) into sparse feature vectors. Each feature column represents a word from the vocabulary and the value denotes the number of times that word appears in a given sample.

Parameters:
#	Param	Default	Type	Description
1	max vocabulary	PHP_INT_MAX	int	The maximum number of words to encode into each word vector.
2	normalize	true	bool	Should we remove extra whitespace and lowercase?
3	tokenizer	Word	object	The object responsible for turning samples of text into individual tokens.
Additional Methods:
Return the fitted vocabulary i.e. the words that will be vectorized:

public vocabulary() : array
Example:
use Rubix\ML\Extractors\CountVectorizer;
use Rubix\ML\Extractors\Tokenizers\Word;

$extractor = new CountVectorizer(5000, true, new Word());

// Return the vocabulary of the vectorizer
$extractor->vocabulary();

// Return the size of the fitted vocabulary
$extractor->size();
Pixel Encoder
Images must first be converted to color channel values in order to be passed to an Estimator. The Pixel Encoder takes an array of images (as PHP Resources) and converts them to a flat vector of color channel data. Image scaling and cropping is handled automatically by Intervention Image. The GD extension is required to use this feature.

Parameters:
#	Param	Default	Type	Description
1	size	[32, 32]	array	A tuple of width and height values denoting the resolution of the encoding.
2	rgb	true	bool	True to use RGB color channel data and False to use Greyscale.
3	sharpen	0	int	A value between 0 and 100 indicating the amount of sharpness to add to each sample.
4	driver	‘gd’	string	The PHP extension to use for image processing (‘gd’ or ‘imagick’).
Additional Methods:
This Extractor does not have any additional methods.

Example:
use Rubix\ML\Extractors\PixelEncoder;

$extractor = new PixelEncoder([28, 28], false, 'imagick');
Estimators
Estimators are the core of the Rubix library and consist of various Classifiers, Regressors, Clusterers, and Anomaly Detectors that make predictions based on their training. Estimators can be supervised or unsupervised depending on the task and can employ methods on top of the basic Estimator API by implementing a number of interfaces such as Online, Probabilistic, and Persistable. They can even be wrapped by a Meta-Estimator to provide additional functionality such as data preprocessing and hyperparameter optimization.

To train an Estimator pass it a training Dataset:

public train(Dataset $training) : void
To make predictions, pass it a new dataset:

public predict(Dataset $dataset) : array
The return value of predict() is an array indexed in the order in which the samples were fed in.

Example:
use Rubix\ML\Classifiers\RandomForest;
use Rubix\ML\Datasets\Labeled;

...
$dataset = new Labeled($samples, $labels);

$estimator = new RandomForest(200, 0.5, 5, 3);

// Take 3 samples out of the dataset to use later
$testing = $dataset->take(3);

// Train the estimator with the labeled dataset
$estimator->train($dataset);

// Make some predictions on the holdout set
$result = $estimator->predict($testing);

var_dump($result);
Output:
array(3) {
	[0] => 'married'
	[1] => 'divorced'
	[2] => 'married'
}
Anomaly Detectors
Anomaly detection is the process of identifying samples that do not conform to an expected pattern. They can be used in fraud prevention, intrusion detection, sciences, and many other areas. The output of a Detector is either 0 for a normal sample or 1 for a detected anomaly.

Isolation Forest
An Ensemble Anomaly Detector comprised of Isolation Trees each trained on a different subset of the training set. The Isolation Forest works by averaging the isolation score of a sample across a user-specified number of trees.

Unsupervised | Probabilistic | Persistable | Nonlinear
Parameters:
#	Param	Default	Type	Description
1	trees	100	int	The number of Isolation Trees to train in the ensemble.
2	ratio	0.1	float	The ratio of random samples to train each Isolation Tree with.
3	threshold	0.5	float	The threshold isolation score. i.e. the probability that a sample is an outlier.
Additional Methods:
This Estimator does not have any additional methods.

Example:
use Rubix\ML\AnomalyDetection\IsolationForest;

$estimator = new IsolationForest(500, 0.1, 0.7);
Isolation Tree
Isolation Trees separate anomalous samples from dense clusters using an extremely randomized splitting process that isolates outliers into their own nodes. Note that this Estimator is considered a weak learner and is typically used within the context of an ensemble (such as Isolation Forest).

Unsupervised | Probabilistic | Persistable | Nonlinear
Parameters:
#	Param	Default	Type	Description
1	max depth	PHP_INT_MAX	int	The maximum depth of a branch that is allowed.
2	threshold	0.5	float	The minimum isolation score. i.e. the probability that a sample is an outlier.
Additional Methods:
This Estimator does not have any additional methods.

Example:
use Rubix\ML\AnomalyDetection\IsolationTree;

$estimator = new IsolationTree(1000, 0.65);
Local Outlier Factor
The Local Outlier Factor (LOF) algorithm considers the local region of a sample, set by the k parameter, when determining an outlier. A density estimate for each neighbor is computed by measuring the radius of the cluster centroid that the point and its neighbors form. The LOF is the ratio of the sample over the median radius of the local region.

Unsupervised | Probabilistic | Online | Persistable | Nonlinear
Parameters:
#	Param	Default	Type	Description
1	k	10	int	The k nearest neighbors that form a local region.
2	neighbors	20	int	The number of neighbors considered when computing the radius of a centroid.
3	threshold	0.5	float	The minimum density score. i.e. the probability that a sample is an outlier.
4	kernel	Euclidean	object	The distance metric used to measure the distance between two sample points.
Additional Methods:
This Estimator does not have any additional methods.

Example:
use Rubix\ML\AnomalyDetection\LocalOutlierFactor;
use Rubix\ML\Kernels\Distance\Minkowski;

$estimator = new LocalOutlierFactor(10, 20, 0.2, new Minkowski(3.5));
Robust Z Score
A quick global anomaly detector, Robust Z Score uses a modified Z score to detect outliers within a Dataset. The modified Z score consists of taking the median and median absolute deviation (MAD) instead of the mean and standard deviation thus making the statistic more robust to training sets that may already contain outliers. Outlier can be flagged in one of two ways. First, their average Z score can be above the user-defined tolerance level or an individual feature’s score could be above the threshold (hard limit).

Unsupervised | Persistable
Parameters:
#	Param	Default	Type	Description
1	tolerance	3.0	float	The average z score to tolerate before a sample is considered an outlier.
2	threshold	3.5	float	The threshold z score of a individual feature to consider the entire sample an outlier.
Additional Methods:
Return the median of each feature column in the training set:

public medians() : array
Return the median absolute deviation (MAD) of each feature column in the training set:

public mads() : array
Example:
use Rubix\ML\AnomalyDetection\RobustZScore;

$estimator = new RobustZScore(1.5, 3.0);
Classifiers
Classifiers are a type of Estimator that predict discrete outcomes such as class labels. There are two types of Classifiers in Rubix - Binary and Multiclass. Binary Classifiers can only distinguish between two classes (ex. Male/Female, Yes/No, etc.) whereas a Multiclass Classifier is able to handle two or more unique class outcomes.

AdaBoost
Short for Adaptive Boosting, this ensemble classifier can improve the performance of an otherwise weak classifier by focusing more attention on samples that are harder to classify.

Supervised | Binary | Persistable
Parameters:
#	Param	Default	Type	Description
1	base	None	string	The fully qualified class name of the base weak classifier.
2	params	[ ]	array	The parameters of the base classifer.
3	estimators	100	int	The number of estimators to train in the ensemble.
4	ratio	0.1	float	The ratio of samples to subsample from the training dataset per epoch.
5	tolerance	1e-3	float	The amount of validation error to tolerate before an early stop is considered.
Additional Methods:
Return the calculated weight values of the last trained dataset:

public weights() : array
Return the influence scores for each boosted classifier:

public influence() : array
Example:
use Rubix\ML\Classifiers\AdaBoost;
use Rubix\ML\Classifiers\ExtraTree;

$estimator = new AdaBoost(ExtraTree::class, [10, 3, 5], 200, 0.1, 1e-2);
Classification Tree
A Decision Tree-based classifier that minimizes gini impurity to greedily search for the best splits in a training set.

Supervised | Multiclass | Probabilistic | Persistable | Nonlinear
Parameters:
#	Param	Default	Type	Description
1	max depth	PHP_INT_MAX	int	The maximum depth of a branch that is allowed. Setting this to 1 is equivalent to training a Decision Stump.
2	min samples	5	int	The minimum number of data points needed to make a prediction.
3	max features	PHP_INT_MAX	int	The maximum number of features to consider when determining a split point.
4	tolerance	1e-3	float	A small amount of impurity to tolerate when choosing a split.
Additional Methods:
This Estimator does not have any additional methods.

Example:
use Rubix\ML\Classifiers\ClassificationTree;

$estimator = new ClassificationTree(100, 7, 4, 1e-4);
Dummy Classifier
A classifier that uses a user-defined Guessing Strategy to make predictions. Dummy Classifier is useful to provide a sanity check and to compare performance with an actual classifier.

Supervised | Multiclass | Persistable
Parameters:
#	Param	Default	Type	Description
1	strategy	PopularityContest	object	The guessing strategy to employ when guessing the outcome of a sample.
Additional Methods:
This Estimator does not have any additional methods.

Example:
use Rubix\ML\Classifiers\DummyClassifier;
use Rubix\ML\Other\Strategies\PopularityContest;

$estimator = new DummyClassifier(new PopularityContest());
Extra Tree
An Extremely Randomized Classification Tree that splits the training set at a random point chosen among the maximum features. Extra Trees work great in Ensembles such as Random Forest or AdaBoost as the “weak” classifier or they can be used on their own. The strength of Extra Trees are computational efficiency as well as increasing variance of the prediction (if that is desired).

Supervised | Multiclass | Probabilistic | Persistable | Nonlinear
Parameters:
#	Param	Default	Type	Description
1	max depth	PHP_INT_MAX	int	The maximum depth of a branch that is allowed. Setting this to 1 is equivalent to training a Decision Stump.
2	min samples	5	int	The minimum number of data points needed to make a prediction.
3	max features	PHP_INT_MAX	int	The number of features to consider when determining a split.
Additional Methods:
This Estimator does not have any additional methods.

Example:
use Rubix\ML\Classifiers\ExtraTree;

$estimator = new ExtraTree(20, 3, 4);
Gaussian Naive Bayes
A variate of the Naive Bayes classifier that uses a probability density function (PDF) over continuous features. The distribution of values is assumed to be Gaussian therefore your data might need to be transformed beforehand if it is not normally distributed.

Supervised | Multiclass | Online | Probabilistic | Persistable | Nonlinear
Parameters:
This Estimator does not have any parameters.

Additional Methods:
Return the class prior log probabilities based on their weight over all training samples:

public priors() : array
Return the running mean of each feature column of the training data:

public means() : array
Return the running variance of each feature column of the training data:

public variances() : array
Example:
use Rubix\ML\Classifiers\GaussianNB;

$estimator = new GaussianNB();
K Nearest Neighbors
A distance-based algorithm that locates the K nearest neighbors from the training set and uses a majority vote to classify the unknown sample. K Nearest Neighbors is considered a lazy learning Estimator because it does the majority of its computation at prediction time.

Supervised | Multiclass | Probabilistic | Online | Persistable | Nonlinear
Parameters:
#	Param	Default	Type	Description
1	k	5	int	The number of neighboring training samples to consider when making a prediction.
2	kernel	Euclidean	object	The distance kernel used to measure the distance between two sample points.
Additional Methods:
This Estimator does not have any additional methods.

Example:
use Rubix\ML\Classifiers\KNearestNeighbors;
use Rubix\ML\Kernels\Distance\Euclidean;

$estimator = new KNearestNeighbors(3, new Euclidean());
Logistic Regression
A type of linear classifier that uses the logistic (sigmoid) function to distinguish between two possible outcomes. Logistic Regression measures the relationship between the class label and one or more independent variables by estimating probabilities.

Supervised | Binary | Online | Probabilistic | Persistable | Linear
Parameters:
#	Param	Default	Type	Description
1	epochs	100	int	The maximum number of training epochs to execute.
2	batch size	10	int	The number of training samples to process at a time.
3	optimizer	Adam	object	The gradient descent optimizer used to train the underlying network.
4	alpha	1e-4	float	The L2 regularization term.
5	min change	1e-8	float	The minimum change in the weights necessary to continue training.
Additional Methods:
Return the training progress at each epoch:

public progress() : array
Return the underlying neural network instance or null if untrained:

public network() : Network|null
Example:
use Rubix\ML\Classifers\LogisticRegression;
use Rubix\ML\NeuralNet\Optimizers\Adam;

$estimator = new LogisticRegression(300, 10, new Adam(0.001), 1e-4, 1e-8);
Multi Layer Perceptron
A multiclass feedforward Neural Network classifier that uses a series of user-defined Hidden Layers as intermediate computational units. Multiple layers and non-linear activation functions allow the Multi Layer Perceptron to handle complex non-linear problems. MLP also features progress monitoring which stops training when it can no longer make progress. It also utilizes snapshotting to make sure that it always uses the best parameters even if progress may have declined during training.

Supervised | Multiclass | Online | Probabilistic | Persistable | Nonlinear
Parameters:
#	Param	Default	Type	Description
1	hidden	[ ]	array	An array of hidden layers of the neural network.
2	batch size	50	int	The number of training samples to process at a time.
3	optimizer	Adam	object	The gradient descent step optimizer used to train the underlying network.
4	alpha	1e-4	float	The L2 regularization term.
5	metric	Accuracy	object	The validation metric used to monitor the training progress of the network.
6	holdout	0.1	float	The ratio of samples to hold out for progress monitoring.
7	window	3	int	The number of epochs to consider when determining if the algorithm should terminate or keep training.
8	tolerance	1e-3	The amount of error to tolerate in the validation metric.
9	epochs	PHP_INT_MAX	int	The maximum number of training epochs to execute.
Additional Methods:
Return the training progress at each epoch:

public progress() : array
Returns the underlying neural network instance or null if untrained:

public network() : Network|null
Example:
use Rubix\ML\Classifiers\MultiLayerPerceptron;
use Rubix\ML\NeuralNet\Layers\Dense;
use Rubix\ML\NeuralNet\ActivationFunctions\ELU;
use Rubix\ML\NeuralNet\Optimizers\Adam;
use Rubix\ML\CrossValidation\Metrics\MCC;

$hidden = [
	new Dense(30, new ELU()),
	new Dense(20, new ELU()),
	new Dense(10, new ELU()),
];

$estimator = new MultiLayerPerceptron($hidden, 100, new Adam(0.001), 1e-4, new MCC(), 0.2, 3, PHP_INT_MAX);
Naive Bayes
Probability-based classifier that used probabilistic inference to predict the correct class. The probabilities are calculated using Bayes Rule. The naive part relates to the fact that it assumes that all features are independent, which is rarely the case in the real world but tends to work out in practice for most problems.

Supervised | Multiclass | Online | Probabilistic | Persistable | Nonlinear
Parameters:
#	Param	Default	Type	Description
1	smoothing	1.0	float	The amount of additive (Laplace) smoothing to apply to the probabilities.
Additional Methods:
Returns the class prior log probabilities based on their weight over all training samples:

public priors() : array
Return the log probabilities of each feature given each class label:

public probabilities() : array
Example:
use Rubix\ML\Classifiers\NaiveBayes;

$estimator = new NaiveBayes(10.0);
Random Forest
Ensemble classifier that trains Decision Trees (Classification Trees or Extra Trees) on a random subset (bootstrap) of the training data. A prediction is made based on the probability scores returned from each tree in the forest weighted equally.

Supervised | Multiclass | Probabilistic | Persistable | Nonlinear
Parameters:
#	Param	Default	Type	Description
1	trees	100	int	The number of Decision Trees to train in the ensemble.
2	ratio	0.1	float	The ratio of random samples to train each Decision Tree with.
3	max depth	10	int	The maximum depth of a branch that is allowed. Setting this to 1 is equivalent to training a Decision Stump.
4	min samples	5	int	The minimum number of data points needed to split a decision node.
5	max features	PHP_INT_MAX	int	The number of features to consider when determining a split.
6	tolerance	1e-3	float	A small amount of Gini impurity to tolerate when choosing a split.
7	base	ClassificationTree::class	string	The base tree class name.
Additional Methods:
This Estimator does not have any additional methods.

Example:
use Rubix\ML\Classifiers\RandomForest;
use Rubix\ML\Classifiers\ClassificationTree;

$estimator = new RandomForest(400, 0.1, 10, 3, 5, 1e-2, ClassificationTree::class);
Softmax Classifier
A generalization of Logistic Regression for multiclass problems using a single layer neural network with a Softmax output layer.

Supervised | Multiclass | Online | Probabilistic | Persistable | Linear
Parameters:
#	Param	Default	Type	Description
1	epochs	100	int	The maximum number of training epochs to execute.
2	batch size	10	int	The number of training samples to process at a time.
3	optimizer	Adam	object	The gradient descent step optimizer used to train the underlying network.
4	alpha	1e-4	float	The L2 regularization term.
5	min change	1e-8	float	The minimum change in the weights necessary to continue training.
Additional Methods:
Return the training progress at each epoch:

public progress() : array
Return the underlying neural network instance or null if untrained:

public network() : Network|null
Example:
use Rubix\ML\Classifiers\SoftmaxClassifier;
use Rubix\ML\NeuralNet\Optimizers\Momentum;

$estimator = new SoftmaxClassifier(300, 100, new Momentum(0.001), 1e-4, 1e-5);
Clusterers
Clustering is a common technique in machine learning that focuses on grouping samples in such a way that the groups are similar. Clusterers take unlabeled data points and assign them a label (cluster). The return value of each prediction is the cluster number each sample was assigned to.

DBSCAN
Density-Based Spatial Clustering of Applications with Noise is a clustering algorithm able to find non-linearly separable and arbitrarily-shaped clusters. In addition, DBSCAN also has the ability to mark outliers as noise and thus can be used as a quasi Anomaly Detector as well.

Unsupervised | Persistable | Nonlinear
Parameters:
#	Param	Default	Type	Description
1	radius	0.5	float	The maximum radius between two points for them to be considered in the same cluster.
2	min density	5	int	The minimum number of points within radius of each other to form a cluster.
3	kernel	Euclidean	object	The distance metric used to measure the distance between two sample points.
Additional Methods:
This Estimator does not have any additional methods.

Example:
use Rubix\ML\Clusterers\DBSCAN;
use Rubix\ML\Kernels\Distance\Diagonal;

$estimator = new DBSCAN(4.0, 5, new Diagonal());
Fuzzy C Means
Distance-based clusterer that allows samples to belong to multiple clusters if they fall within a fuzzy region defined by the fuzz parameter. Fuzzy C Means is similar to both K Means and Gaussian Mixture Models in that they require apriori knowledge of the number (parameter c) of clusters.

Unsupervised | Probabilistic | Persistable
Parameters:
#	Param	Default	Type	Description
1	c	None	int	The number of target clusters.
2	fuzz	2.0	float	Determines the bandwidth of the fuzzy area.
3	kernel	Euclidean	object	The distance metric used to measure the distance between two sample points.
4	threshold	1e-4	float	The minimum change in centroid means necessary for the algorithm to continue training.
5	epochs	PHP_INT_MAX	int	The maximum number of training rounds to execute.
Additional Methods:
Return the c computed centroids of the training data:

public centroids() : array
Returns the progress of training at each epoch:

public progress() : array
Example:
use Rubix\ML\Clusterers\FuzzyCMeans;
use Rubix\ML\Kernels\Distance\Euclidean;

$estimator = new FuzzyCMeans(5, 1.2, new Euclidean(), 1e-3, 1000);
K Means
A fast centroid-based hard clustering algorithm capable of clustering linearly separable data points given a number of target clusters set by the parameter K.

Unsupervised | Online | Persistable | Linear
Parameters:
#	Param	Default	Type	Description
1	k	None	int	The number of target clusters.
2	kernel	Euclidean	object	The distance metric used to measure the distance between two sample points.
3	epochs	PHP_INT_MAX	int	The maximum number of training rounds to execute.
Additional Methods:
Return the c computed centroids of the training data:

public centroids() : array
Example:
use Rubix\ML\Clusterers\KMeans;
use Rubix\ML\Kernels\Distance\Euclidean;

$estimator = new KMeans(3, new Euclidean());
Mean Shift
A hierarchical clustering algorithm that uses peak finding to locate the local maxima (centroids) of a training set given by a radius constraint.

Unsupervised | Persistable | Linear
Parameters:
#	Param	Default	Type	Description
1	radius	None	float	The radius of each cluster centroid.
2	kernel	Euclidean	object	The distance metric used to measure the distance between two sample points.
3	threshold	1e-8	float	The minimum change in centroid means necessary for the algorithm to continue training.
4	epochs	PHP_INT_MAX	int	The maximum number of training rounds to execute.
Additional Methods:
Return the c computed centroids of the training data:

public centroids() : array
Returns the progress of training at each epoch:

public progress() : array
Example:
use Rubix\ML\Clusterers\MeanShift;
use Rubix\ML\Kernels\Distance\Diagonal;

$estimator = new MeanShift(3.0, new Diagonal(), 1e-6, 2000);
Regressors
Regression analysis is used to predict the outcome of an event where the value is continuous.

Dummy Regressor
Regressor that guesses the output values based on a Guessing Strategy. Dummy Regressor is useful to provide a sanity check and to compare performance against actual Regressors.

Supervised | Persistable
Parameters:
#	Param	Default	Type	Description
1	strategy	BlurryMean	object	The guessing strategy to employ when guessing the outcome of a sample.
Additional Methods:
This Estimator does not have any additional methods.

Example:
use Rubix\ML\Regressors\DummyRegressor;
use Rubix\ML\Other\Strategies\BlurryMedian;

$estimator = new DummyRegressor(new BlurryMedian(0.2));
KNN Regressor
A version of K Nearest Neighbors that uses the mean outcome of K nearest data points to make continuous valued predictions suitable for regression problems.

Supervised | Online | Persistable | Nonlinear
Parameters:
#	Param	Default	Type	Description
1	k	5	int	The number of neighboring training samples to consider when making a prediction.
2	kernel	Euclidean	object	The distance kernel used to measure the distance between two sample points.
Additional Methods:
This Estimator does not have any additional methods.

Example:
use Rubix\ML\Regressors\KNNRegressor;
use Rubix\ML\Kernels\Distance\Minkowski;

$estimator = new KNNRegressor(2, new Minkowski(3.0));
MLP Regressor
A Neural Network with a linear output layer suitable for regression problems. The MLP also features progress monitoring which stops training when it can no longer make progress. It also utilizes snapshotting to make sure that it always uses the best parameters even if progress declined during training.

Supervised | Persistable | Nonlinear
Parameters:
#	Param	Default	Type	Description
1	hidden	[ ]	array	An array of hidden layers of the neural network.
2	batch size	10	int	The number of training samples to process at a time.
3	optimizer	Adam	object	The gradient descent step optimizer used to train the underlying network.
4	alpha	1e-4	float	The L2 regularization term.
5	metric	Accuracy	object	The validation metric used to monitor the training progress of the network.
6	holdout	0.1	float	The ratio of samples to hold out for progress monitoring.
7	window	3	int	The number of epochs to consider when determining if the algorithm should terminate or keep training.
8	tolerance	float	1e-5	The amount of error to tolerate in the validation metric.
9	epochs	PHP_INT_MAX	int	The maximum number of training epochs to execute.
Additional Methods:
Return the training progress at each epoch:

public progress() : array
Returns the underlying neural network instance or null if untrained:

public network() : Network|null
Example:
use Rubix\ML\Regressors\MLPRegressor;
use Rubix\ML\NeuralNet\Layers\Dense;
use Rubix\ML\NeuralNet\ActivationFunctions\ISRU;
use Rubix\ML\NeuralNet\Optimizers\RMSProp;
use Rubix\ML\CrossValidation\Metrics\MeanSquaredError;

$hidden = [
	new Dense(100, new ISRU()),
];

$estimator = new MLPRegressor($hidden, 50, new RMSProp(0.001), 1e-2, new MeanSquaredError(), 0.1, 3, PHP_INT_MAX);
Regression Tree
A Decision Tree learning algorithm that performs greedy splitting by minimizing the sum of squared errors between decision node splits.

Supervised | Persistable | Nonlinear
Parameters:
#	Param	Default	Type	Description
1	max depth	PHP_INT_MAX	int	The maximum depth of a branch that is allowed.
2	min samples	5	int	The minimum number of data points needed to make a prediction.
3	max features	PHP_INT_MAX	int	The maximum number of features to consider when determining a split point.
4	tolerance	1e-4	float	A small amount of impurity to tolerate when choosing a split.
Additional Methods:
This Estimator does not have any additional methods.

Example:
use Rubix\ML\Regressors\RegressionTree;

$estimator = new RegressionTree(80, 1, 10, 0.0);
Ridge
L2 penalized least squares linear regression. Can be used for simple regression problems that can be fit to a straight line.

Supervised | Persistable | Linear
Parameters:
#	Param	Default	Type	Description
1	alpha	1.0	float	The L2 regularization term.
Additional Methods:
Return the y intercept of the computed regression line:

public intercept() : float|null
Return the computed coefficients of the regression line:

public coefficients() : array
Example:
use Rubix\ML\Regressors\Ridge;

$estimator = new Ridge(2.0);
Estimator Interfaces
Online
Certain Estimators that implement the Online interface can be trained in batches. Estimators of this type are great for when you either have a continuous stream of data or a dataset that is too large to fit into memory. Partial training allows the model to grow as new data is acquired.

You can partially train an Online Estimator with:

public partial(Dataset $dataset) : void
Example:
...
$datasets = $dataset->fold(3);

$estimator->partial($dataset[0]);

$estimator->partial($dataset[1]);

$estimator->partial($dataset[2]);
It is important to note that an Estimator will continue to train as long as you are using the partial() method, however, calling train() on a trained or partially trained Estimator will reset it back to baseline first.

Probabilistic
Some Estimators may implement the Probabilistic interface, in which case, they will have an additional method that returns an array of probability scores of each possible class, cluster, etc. Probabilities are useful for ascertaining the degree to which the Estimator is certain about a particular outcome.

Calculate probability estimates:

public proba(Dataset $dataset) : array
Example:
...
$probabilities = $estimator->proba($dataset->head(2));  

var_dump($probabilities);
Output:
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
Meta-Estimators
Meta-Estimators allow you to progressively enhance your models by adding additional functionality such as data preprocessing and persistence or by orchestrating an Ensemble of base Estimators. Each Meta-Estimator wraps a base Estimator and you can even wrap certain Meta-Estimators with other Meta-Estimators. Some examples of Meta-Estimators in Rubix are Pipeline, Grid Search, and Bootstrap Aggregator.

Example:
use Rubix\ML\Pipeline;
use Rubix\ML\GridSearch;
use Rubix\ML\Classifiers\ClassificationTree;
use Rubix\ML\CrossValidation\Metrics\MCC;
use Rubix\ML\CrossValidation\KFold;
use Rubix\ML\Transformers\NumericStringConverter;

...
$params = [[10, 30, 50], [1, 3, 5], [2, 3, 4];

$estimator = new Pipeline(new GridSearch(ClassificationTree::class, $params, new MCC(), new KFold(10)));

$estimator->train($dataset); // Train a classification tree with preprocessing and grid search

$estimator->complexity(); // Call complexity() method on Decision Tree from Pipeline
Data Preprocessing
Often, additional processing of input data is required to deliver correct predictions and/or accelerate the training process. In this section, we’ll introduce the Pipeline meta-Estimator and the various Transformers that it employs to fit the input data to suit the requirements and preferences of the Estimator that it feeds.

Pipeline
Pipeline is responsible for transforming the input sample matrix of a Dataset in such a way that can be processed by the base Estimator. Pipeline accepts a base Estimator and a list of Transformers to apply to the input data before it is fed to the learning algorithm. Under the hood, Pipeline will automatically fit the training set upon training and transform any Dataset object supplied as an argument to one of the base Estimator’s methods, including predict().

Classifiers, Regressors, Clusterers, Anomaly Detectors
Parameters:
#	Param	Default	Type	Description
1	estimator	None	object	An instance of a base estimator.
2	transformers	[ ]	array	The transformer middleware to be applied to each dataset.
Additional Methods:
This Meta Estimator does not have any additional methods.

Example:
use Rubix\ML\Pipeline;
use Rubix\ML\Classifiers\SoftmaxClassifier;
use Rubix\ML\NeuralNet\Optimizer\RMSProp;
use Rubix\ML\Transformers\MissingDataImputer;
use Rubix\ML\Transformers\OneHotEncoder;
use Rubix\ML\Transformers\SparseRandomProjector;

$estimator = new Pipeline(new SoftmaxClassifier(100, new RMSProp(0.01), 1e-2), [
	new MissingDataImputer(),
	new OneHotEncoder(),
	new SparseRandomProjector(30),
]);

$estimator->train($dataset); // Datasets are fit and ...

$estimator->predict($samples); // Transformed automatically.
Transformer middleware will process in the order given when the Pipeline was built and cannot be reordered without instantiating a new one. Since Tranformers run sequentially, the order in which they run matters. For example, a Transformer near the end of the stack may depend on a previous Transformer to convert all categorical features into continuous ones before it can run.

In practice, applying transformations can drastically improve the performance of your model by cleaning, scaling, expanding, compressing, and normalizing the input data.

Ensemble
Ensemble Meta Estimators train and orchestrate a number of base Estimators in order to make their predictions. Certain Estimators (like AdaBoost and Random Forest) are implemented as Ensembles under the hood, however these Meta Estimators are able to work across Estimator types which makes them very useful.

Bootstrap Aggregator
Bootstrap Aggregating (or bagging) is a model averaging technique designed to improve the stability and performance of a user-specified base Estimator by training a number of them on a unique bootstrapped training set. Bootstrap Aggregator then collects all of their predictions and makes a final prediction based on the results.

Classifiers, Regressors, Anomaly Detectors
Parameters:
#	Param	Default	Type	Description
1	base	None	string	The fully qualified class name of the base Estimator.
2	params	[ ]	array	The parameters of the base estimator.
3	estimators	10	int	The number of base estimators to train in the ensemble.
4	ratio	0.5	float	The ratio of random samples to train each estimator with.
Additional Methods:
This Meta Estimator does not have any additional methods.

Example:
use Rubix\ML\BootstrapAggregator;
use Rubix\ML\Regressors\RegressionTree;

...
$estimator = new BootstrapAggregator(RegressionTree::class, [10, 5, 3], 100, 0.2);

$estimator->traing($training); // Trains 100 regression trees

$estimator->predict($testing); // Aggregates their predictions
Committee Machine
A voting Ensemble that aggregates the predictions of a committee of user-specified, heterogeneous estimators (called experts) of a single type (i.e all Classifiers, Regressors, etc). The committee uses a hard-voting scheme to make final predictions.

Classifiers, Regressors, Anomaly Detectors
Parameters:
#	Param	Default	Type	Description
1	experts	[ ]	array	An array of estimator instances.
Additional Methods:
This Meta Estimator does not have any additional methods.

Example:
use Rubix\ML\Classifiers\CommitteeMachine;
use Rubix\ML\Classifiers\RandomForest;
use Rubix\ML\Classifiers\SoftmaxClassifier;
use Rubix\ML\NeuralNet\Optimizers\Adam;
use Rubix\ML\Classifiers\KNearestNeighbors;

$estimator = new CommitteeMachine([
	new RandomForest(100, 0.3, 30, 3, 4, 1e-3),
	new SoftmaxClassifier(50, new Adam(0.001), 0.1),
	new KNearestNeighbors(3),
]);
Model Selection
Model selection is the task of selecting a version of a model with a hyperparameter combination that maximizes performance on a specific validation metric. Rubix provides the Grid Search meta-Estimator that performs an exhaustive search over all combinations of parameters given as possible arguments.

Grid Search
Grid Search is an algorithm that optimizes hyperparameter selection. From the user’s perspective, the process of training and predicting is the same, however, under the hood, Grid Search trains one Estimator per combination of parameters and predictions are made using the best Estimator. You can access the scores for each parameter combination by calling the results() method on the trained Grid Search meta-Estimator or you can get the best parameters by calling best().

Parameters:
#	Param	Default	Type	Description
1	base	None	string	The fully qualified class name of the base Estimator.
2	params	[ ]	array	An array containing n-tuples of parameters where each tuple represents a possible parameter for a given parameter location (ordinal).
3	metric	None	object	The validation metric used to score each set of parameters.
4	validator	None	object	An instance of a Validator object (HoldOut, KFold, etc.) that will be used to test each parameter combination.
Additional Methods:
Return the results (scores and parameters) of the last search:

public results() : array
Return the he parameters with the highest validation score:

public best() : array
Return the underlying estimator trained with the best parameters:

public estimator() : Estimator
Example:
use Rubix\ML\GridSearch;
use Rubix\ML\Classifiers\KNearestNeighbors;
use Rubix\ML\Kernels\Distance\Euclidean;
use Rubix\ML\Kernels\Distance\Manhattan;
use Rubix\ML\CrossValidation\Metrics\Accuracy;
use Rubix\ML\CrossValidation\KFold;

...
$params = [
	[1, 3, 5, 10], [new Euclidean(), new Manhattan()],
];

$estimator = new GridSearch(KNearestNeightbors::class, $params, new Accuracy(), new KFold(10));

$estimator->train($dataset); // Train one estimator per parameter combination

var_dump($estimator->best()); // Return the best combination
Output:
array(2) {
  [0]=> int(5)
  [1]=> object(Rubix\ML\Kernels\Distance\Euclidean)#15 (0) {
  }
}
Random Search
Random search is a hyperparameter selection technique that samples n parameters randomly from a user-specified distribution. In Rubix, the Random Params helper can be used along with Grid Search to achieve the goal of random search. The Random Params helper automatically takes care of deduplication so you never need to worry about testing a parameter twice. For this reason, however, you cannot generate more parameters than in range of, thus generating 5 unique ints between 1 and 3 is impossible.

To generate a distribution of integer parameters:

public static ints(int $min, int $max, int $n = 10) : array
To generate a distribution of floating point parameters:

public static floats(float $min, float $max, int $n = 10) : array
Example:
use Rubix\ML\GridSearch;
use Rubix\ML\Other\Helpers\RandomParams;
use Rubix\ML\Clusterers\FuzzyCMeans;
use Rubix\ML\Kernels\Distance\Diagonal;
use Rubix\ML\Kernels\Distance\Minkowski;
use Rubix\CrossValidation\KFold;
use Rubix\CrossValidation\Metrics\VMeasure;

...
$params = [
	[1, 2, 3, 4, 5], RandomParams::floats(1.0, 20.0, 20), [new Diagonal(), new Minkowski(3.0)],
];

$estimator = new GridSearch(FuzzyCMeans::class, $params, new VMeasure(), new KFold(10));

$estimator->train($dataset);

var_dump($estimator->best());
Output:
array(3) {
  [0]=> int(4)
  [1]=> float(13.65)
  [2]=> object(Rubix\ML\Kernels\Distance\Diagonal)#15 (0) {
  }
}
Model Persistence
Model persistence is the practice of saving a trained model to disk so that it can be restored later, on a different machine, or used in an online system. Rubix persists your models using built in PHP object serialization (similar to pickling in Python). Most Estimators are persistable, but some are not allowed due to their poor storage complexity.

Persistent Model
It is possible to persist a model to disk by wrapping the Estimator instance in a Persistent Model meta-Estimator. The Persistent Model class gives the Estimator two additional methods save() and restore() that serialize and unserialize to and from disk. In order to be persisted the Estimator must implement the Persistable interface.

public save(string $path) : bool
Where path is the location of the directory where you want the model saved. save() will return true if the model was successfully persisted and false if it failed.

public static restore(string $path) : self
The restore method will return an instantiated model from the save path.

Example:
use Rubix\ML\PersistentModel;
use Rubix\ML\Classifiers\RandomForest;

$estimator = new PersistentModel(new RandomForest(100, 0.2, 10, 3));

$estimator->save('path/to/models/folder/random_forest.model');

$estimator->save(); // Saves to current working directory under unique filename

$estimator = PersistentModel::restore('path/to/models/folder/random_forest.model');
Transformers
Transformers take a sample matrices and transform them in various ways. A common transformation is scaling and centering the values using one of the Standardizers (Z Scale, Robust, Quartile). Transformers can be used with the Pipeline meta-Estimator or they can be used separately.

The fit method will allow the transformer to compute any necessary information from the training set in order to carry out its transformations. You can think of fitting a Transformer like training an Estimator. Not all Transformers need to be fit to the training set, when in doubt, call fit() anyways.

public fit(Dataset $dataset) : void
The Transformer directly modifies a sample matrix via the transform() method.

public transform(array $samples) : void
To transform a Dataset without having to pass the raw sample matrix you can call apply() on any Dataset object and it will apply the transformation to the underlying sample matrix automatically.

Example:
use Rubix\ML\Transformers\MinMaxNormalizer;

...
$transformer = new MinMaxNormalizer();

$dataset->apply($transformer);
Here are a list of the Transformers available in Rubix.

Dense and Sparse Random Projectors
A Random Projector is a dimensionality reducer based on the Johnson-Lindenstrauss lemma that uses a random matrix to project a feature vector onto a user-specified number of dimensions. It is faster than most non-randomized dimensionality reduction techniques and offers similar performance.

The difference between the Dense and Sparse Random Projectors are that the Dense version uses a dense random guassian distribution and the Sparse version uses a sparse matrix (mostly 0’s).

Continuous Only
Parameters:
#	Param	Default	Type	Description
1	dimensions	None	int	The number of target dimensions to project onto.
Additional Methods:
This Transformer does not have any additional methods.

Example:
use Rubix\ML\Transformers\DenseRandomProjector;
use Rubix\ML\Transformers\SparseRandomProjector;

$transformer = new DenseRandomProjector(50);

$transformer = new SparseRandomProjector(50);
L1 and L2 Regularizers
Augment each sample vector in the sample matrix such that each feature is divided over the L1 or L2 norm (or magnitude) of that vector.

Continuous Only
Parameters:
This Transformer does not have any parameters.

Additional Methods:
This Transformer does not have any additional methods.

Example:
use Rubix\ML\Transformers\L1Regularizer;
use Rubix\ML\Transformers\L2Regularizer;

$transformer = new L1Regularizer();
$transformer = new L2Regularizer();
Lambda Function
Run a stateless lambda function (anonymous function) over the sample matrix. The lambda function receives the sample matrix as an argument and should return the transformed sample matrix.

Categorical or Continuous
Parameters:
#	Param	Default	Type	Description
1	lambda	None	callable	The lambda function to run over the sample matrix.
Additional Methods:
This Transformer does not have any additional methods.

Example:
use Rubix\ML\Transformers\LambdaFunction;

// Instantiate a lambda function that will sum up all the features for each sample
$transformer = new LambdaFunction(function ($samples) {
	return array_map(function ($sample) {
		return [array_sum($sample)];
	}, $samples);
});
Min Max Normalizer
Often used as an alternative to Standard Scaling, the Min Max Normalization scales the input features to a range of between 0 and 1 by dividing the feature value over the maximum value for that feature column.

Continuous
Parameters:
This Transformer does not have any parameters.

Additional Methods:
This Transformer does not have any additional methods.

Example:
use Rubix\ML\Transformers\MinMaxNormalizer;

$transformer = new MinMaxNormalizer();
Missing Data Imputer
In the real world, it is common to have data with missing values here and there. The Missing Data Imputer replaces missing value placeholders with a guess based on a given guessing Strategy.

Categorical or Continuous
Parameters:
#	Param	Default	Type	Description
1	placeholder	‘?’	string or numeric	The placeholder that denotes a missing value.
2	continuous strategy	BlurryMean	object	The guessing strategy to employ for continuous feature columns.
3	categorical strategy	PopularityContest	object	The guessing strategy to employ for categorical feature columns.
Additional Methods:
This Transformer does not have any additional methods.

Example:
use Rubix\ML\Transformers\MissingDataImputer;
use Rubix\ML\Transformers\Strategies\BlurryMean;
use Rubix\ML\Transformers\Strategies\PopularityContest;

$transformer = new MissingDataImputer('?', new BlurryMean(0.2), new PopularityContest());
Numeric String Converter
This handy Transformer will convert all numeric strings into their floating point counterparts. Useful for when extracting from a source that only recognizes data as string types.

Categorical
Parameters:
This Transformer does not have any parameters.

Additional Methods:
This Transformer does not have any additional methods.

Example:
use Rubix\ML\Transformers\NumericStringConverter;

$transformer = new NumericStringConverter();
One Hot Encoder
The One Hot Encoder takes a column of categorical features and produces a one-hot vector of n-dimensions where n is equal to the number of unique categories per feature column. This is used when you need to convert all features to continuous format since some Estimators do not work with categorical features.

Categorical
Parameters:
#	Param	Default	Type	Description
1	columns	Null	array	The user-specified columns to encode indicated by numeric index starting at 0.
Additional Methods:
This Transformer does not have any additional methods.

Example:
use Rubix\ML\Transformers\OneHotEncoder;

$transformer = new OneHotEncoder([0, 3, 5, 7, 9]);
Polynomial Expander
This Transformer will generate polynomial features up to and including the specified degree. Polynomial expansion is often used to fit data that is non-linear using a linear Estimator such as Ridge.

Continuous Only
Parameters:
#	Param	Default	Type	Description
1	degree	2	int	The highest degree polynomial to generate from each feature vector.
Additional Methods:
This Transformer does not have any additional methods.

Example:
use Rubix\ML\Transformers\PolynomialExpander;

$transformer = new PolynomialExpander(3);
Quartile Standardizer
This standardizer removes the median and scales each sample according to the interquantile range (IQR). The IQR is the range between the 1st quartile (25th quantile) and the 3rd quartile (75th quantile).

Continuous
Parameters:
This Transformer does not have any parameters.

Additional Methods:
Return the medians calculated by fitting the training set:

public medians() : array
Return the interquartile ranges calculated during fitting:

public iqrs() : array
Example:
use Rubix\ML\Transformers\QuartileStandardizer;

$transformer = new QuartileStandardizer();
Robust Standardizer
This Transformer standardizes continuous features by removing the median and dividing over the median absolute deviation (MAD), a value referred to as robust z score. The use of robust statistics makes this standardizer more immune to outliers than the Z Scale Standardizer.

Continuous
Parameters:
This Transformer does not have any parameters.

Additional Methods:
Return the medians calculated by fitting the training set:

public medians() : array
Return the median absolute deviations calculated during fitting:

public mads() : array
Example:
use Rubix\ML\Transformers\RobustStandardizer;

$transformer = new RobustStandardizer();
TF-IDF Transformer
Term Frequency - Inverse Document Frequency is a measure of how important a word is to a document. The TF-IDF value increases proportionally with the number of times a word appears in a document and is offset by the frequency of the word in the corpus. This Transformer makes the assumption that the input is made up of word frequency vectors such as those created by the Count Vectorizer.

Continuous Only
Parameters:
This Transformer does not have any parameters.

Additional Methods:
This Transformer does not have any additional methods.

Example:
$transformer = new TfIdfTransformer();
Variance Threshold Filter
A type of feature selector that removes all columns that have a lower variance than the threshold. Variance is computed as the population variance of all the values in the feature column.

Categorical and Continuous
Parameters:
#	Param	Default	Type	Description
1	threshold	0.0	float	The threshold at which lower scoring columns will be dropped from the dataset.
Additional Methods:
Return the columns that were selected during fitting:

public selected() : array
Example:
use Rubix\ML\Transformers\VarianceThresholdFilter;

$transformer = new VarianceThresholdFilter(50);
Z Scale Standardizer
A way of centering and scaling an input vector by computing the Z Score for each continuous feature.

Continuous
Parameters:
This Transformer does not have any parameters.

Additional Methods:
Return the means calculated by fitting the training set:

public means() : array
Return the standard deviations calculated during fitting:

public stddevs() : array
Example:
use Rubix\ML\Transformers\ZScaleStandardizer;

$transformer = new ZScaleStandardizer();
Neural Network
A number of the Estimators in Rubix are implemented as a computational graph commonly referred to as a Neural Network due to its inspiration from the human brain. Neural Nets are trained using an iterative process called Gradient Descent and use Backpropagation (sometimes called Reverse Mode Autodiff) to calculate the error of each parameter in the network.

The Multi Layer Perceptron and MLP Regressor are both neural networks capable of being built with an almost limitless combination of Hidden layers employing various Activation Functions. The strength of deep neural nets (with 1 or more hidden layers) is its diversity in handling large amounts of data. In general, the deeper the neural network, the better it will perform.

Activation Functions
The input to every neuron is passed through an Activation Function which determines its output. There are different properties of Activation Functions that make them more or less desirable depending on your problem.

ELU
Exponential Linear Units are a type of rectifier that soften the transition from non-activated to activated using the exponential function.

Parameters:
#	Param	Default	Type	Description
1	alpha	1.0	float	The value at which leakage will begin to saturate. Ex. alpha = 1.0 means that the output will never be more than -1.0 when inactivated.
Example:
use Rubix\ML\NeuralNet\ActivationFunctions\ELU;

$activationFunction = new ELU(5.0);
Hyperbolic Tangent
S-shaped function that squeezes the input value into an output space between -1 and 1 centered at 0.

Parameters:
This Activation Function does not have any parameters.

Example:
use Rubix\ML\NeuralNet\ActivationFunctions\HyperbolicTangent;

$activationFunction = new HyperbolicTangent();
Identity
The Identity function (sometimes called Linear Activation Function) simply outputs the value of the input.

Parameters:
This Activation Function does not have any parameters.

Example:
use Rubix\ML\NeuralNet\ActivationFunctions\Identity;

$activationFunction = new Identity();
ISRU
Inverse Square Root units have a curve similar to Hyperbolic Tangent and Sigmoid but use the inverse of the square root function instead. It is purported by the authors to be computationally less complex than either of the aforementioned. In addition, ISRU allows the parameter alpha to control the range of activation such that it equals + or - 1 / sqrt(alpha).

Parameters:
#	Param	Default	Type	Description
1	alpha	1.0	float	The parameter that controls the range of activation.
Example:
use Rubix\ML\NeuralNet\ActivationFunctions\ISRU;

$activationFunction = new ISRU(2.0);
Leaky ReLU
Leaky Rectified Linear Units are functions that output x when x > 0 or a small leakage value when x < 0. The amount of leakage is controlled by the user-specified parameter.

Parameters:
#	Param	Default	Type	Description
1	leakage	0.01	float	The amount of leakage as a ratio of the input value.
Example:
use Rubix\ML\NeuralNet\ActivationFunctions\LeakyReLU;

$activationFunction = new LeakyReLU(0.001);
SELU
Scaled Exponential Linear Unit is a self-normalizing activation function based on ELU.

Parameters:
#	Param	Default	Type	Description
1	scale	1.05070	float	The factor to scale the output by.
2	alpha	1.67326	float	The value at which leakage will begin to saturate. Ex. alpha = 1.0 means that the output will never be more than -1.0 when inactivated.
Example:
use Rubix\ML\NeuralNet\ActivationFunctions\SELU;

$activationFunction = new SELU(1.05070, 1.67326);
Sigmoid
A bounded S-shaped function (specifically the Logistic function) with an output value between 0 and 1.

Parameters:
This Activation Function does not have any parameters.

Example:
use Rubix\ML\NeuralNet\ActivationFunctions\Sigmoid;

$activationFunction = new Sigmoid();
Soft Plus
A smooth approximation of the ReLU function whose output is constrained to be positive.

Parameters:
This Activation Function does not have any parameters.

Example:
use Rubix\ML\NeuralNet\ActivationFunctions\SoftPlus;

$activationFunction = new SoftPlus();
Softsign
A function that squashes the output of a neuron to + or - 1 from 0. In other words, the output is between -1 and 1.

Parameters:
This Activation Function does not have any parameters.

Example:
use Rubix\ML\NeuralNet\ActivationFunctions\Softsign;

$activationFunction = new Softsign();
Layers
Every network is made up of layers of computational units called neurons. Each layer processes and transforms the input from the previous layer.

There are three types of Layers that form a network, Input, Hidden, and Output. A network can have as many Hidden layers as the user specifies, however, there can only be 1 Input and 1 Output layer per network.

Example:
use Rubix\ML\NeuralNet\Network;
use Rubix\ML\NeuralNet\Layers\Input;
use Rubix\ML\NeuralNet\Layers\Dense;
use Rubix\ML\NeuralNet\Layers\Softmax;
use Rubix\ML\NeuralNet\ActivationFunctions\ELU;
use Rubix\ML\NeuralNet\Optimizers\Adam;

$network = new Network(new Input(784), [
	new Dense(100, new ELU()),
	new Dense(100, new ELU()),
	new Dense(100, new ELU()),
], new Softmax([
	'dog', 'cat', 'frog', 'car',
], 1e-4), new Adam(0.001));
Input
The Input Layer is simply a placeholder layer that represents the value of a sample or batch of samples. The number of placeholder nodes should be equal to the number of feature columns of a sample.

Hidden
In multilayer networks, Hidden layers perform the bulk of the computation. They are responsible for transforming the input space in such a way that can be linearly separable by the Output layer. The more complex the problem space is, the more Hidden layers and neurons will be necessary to handle the complexity.

Dense
Dense layers are fully connected Hidden layers, meaning each neuron is connected to each other neuron in the previous layer. Dense layers are able to employ a variety of Activation Functions that modify the output of each neuron in the layer.

Parameters:
#	Param	Default	Type	Description
1	neurons	None	int	The number of neurons in the layer.
2	activation fn	None	object	The activation function to use.
Example:
use Rubix\ML\NeuralNet\Layers\Dense;
use Rubix\ML\NeuralNet\ActivationFunctions\LeakyReLU;

$layer = new Dense(100, new LeakyReLU(0.05));
Output
Activations are read directly from the Output layer when it comes to making a prediction. The type of Output layer used will determine the type of Estimator the Neural Net can power (Binary Classifier, Multiclass Classifier, or Regressor). The different types of Output layers are listed below.

Linear
The Linear Output Layer consists of a single linear neuron that outputs a continuous scalar value useful for Regression problems.

Parameters:
#	Param	Default	Type	Description
1	alpha	1e-4	float	The L2 regularization penalty.
Example:
use Rubix\ML\NeuralNet\Layers\Linear;

$layer = new Linear(1e-5);
Logit
This Logit layer consists of a single Sigmoid neuron capable of distinguishing between two classes. The Logit layer is useful for neural networks that output a binary class prediction.

Parameters:
#	Param	Default	Type	Description
1	classes	None	array	The unique class labels of the binary classification problem.
2	alpha	1e-4	float	The L2 regularization penalty.
Example:
use Rubix\ML\NeuralNet\Layers\Logit;

$layer = new Logit(['yes', 'no'], 1e-5);
Softmax
A generalization of the Logistic Layer, the Softmax Output Layer gives a joint probability estimate of a multiclass classification problem.

Parameters:
#	Param	Default	Type	Description
1	classes	None	array	The unique class labels of the multiclass classification problem.
2	alpha	1e-4	float	The L2 regularization penalty.
Example:
use Rubix\ML\NeuralNet\Layers\Softmax;

$layer = new Softmax(['yes', 'no', 'maybe'], 1e-6);
Optimizers
Gradient Descent is an algorithm that takes iterative steps towards minimizing an objective function. There have been many papers that describe enhancements to the standard Stochastic Gradient Descent algorithm whose methods are encapsulated in pluggable Optimizers by Rubix. More specifically, Optimizers control the amount of Gradient Descent step to take for each parameter in the network upon each training iteration.

AdaGrad
Short for Adaptive Gradient, the AdaGrad Optimizer speeds up the learning of parameters that do not change often and slows down the learning of parameters that do enjoy heavy activity.

Parameters:
#	Param	Default	Type	Description
1	rate	0.001	float	The learning rate. i.e. the master step size.
Example:
use Rubix\ML\NeuralNet\Optimizers\AdaGrad;

$optimizer = new AdaGrad(0.035);
Adam
Short for Adaptive Momentum Estimation, the Adam Optimizer uses both Momentum and RMS properties to achieve a balance of velocity and stability.

Parameters:
#	Param	Default	Type	Description
1	rate	0.001	float	The learning rate. i.e. the master step size.
2	momentum	0.9	float	The decay rate of the Momentum property.
3	rms	0.999	float	The decay rate of the RMS property.
4	epsilon	1e-8	float	The smoothing constant used for numerical stability.
Example:
use Rubix\ML\NeuralNet\Optimizers\Adam;

$optimizer = new Adam(0.0001, 0.9, 0.999, 1e-8);
Momentum
Momentum adds velocity to each step until exhausted. It does so by accumulating momentum from past updates and adding a factor to the current step.

Parameters:
#	Param	Default	Type	Description
1	rate	0.001	float	The learning rate. i.e. the master step size.
2	decay	0.9	float	The Momentum decay rate.
Example:
use Rubix\ML\NeuralNet\Optimizers\Momentum;

$optimizer = new Momentum(0.001, 0.925);
RMS Prop
An adaptive gradient technique that divides the current gradient over a rolling window of magnitudes of recent gradients.

Parameters:
#	Param	Default	Type	Description
1	rate	0.001	float	The learning rate. i.e. the master step size.
2	decay	0.9	float	The RMS decay rate.
Example:
use Rubix\ML\NeuralNet\Optimizers\RMSProp;

$optimizer = new RMSProp(0.01, 0.9);
Step Decay
A learning rate decay stochastic optimizer that reduces the learning rate by a factor of the decay parameter when it reaches a new floor (takes k steps).

Parameters:
#	Param	Default	Type	Description
1	rate	0.001	float	The learning rate. i.e. the master step size.
2	k	10	int	The size of every floor in steps. i.e. the number of steps to take before applying another factor of decay.
3	decay	1e-5	float	The decay factor to decrease the learning rate by every k steps.
Example:
use Rubix\ML\NeuralNet\Optimizers\StepDecay;

$optimizer = new StepDecay(0.001, 15, 1e-5);
Stochastic
A constant learning rate Optimizer.

Parameters:
#	Param	Default	Type	Description
1	rate	0.001	float	The learning rate. i.e. the master step size.
Example:
use Rubix\ML\NeuralNet\Optimizers\Stochastic;

$optimizer = new Stochastic(0.001);
Snapshots
Snapshots are a way to capture the state of a neural network at a moment in time. A Snapshot object holds all of the parameters in the network and can be used to restore the network back to a previous state.

To take a snapshot of your network simply call the read() method on the Network object. To restore the network from a snapshot pass the snapshot to the restore() method.

The example below shows how to take a snapshot and then restore the network via the snapshot.

...
$snapshot = $network->read();

...

$network->restore($snapshot);
...
Kernel Functions
Kernel functions are used to compute the similarity or distance between two vectors and can be plugged in to a particular Estimator to perform a part of the computation. They are pairwise positive semi-definite functions meaning their output is always 0 or greater. When considered as a hyperparameter, different Kernel functions have properties that can lead to different training and predictions.

Distance
Distance functions are a type of Kernel that measures the distance between two coordinate vectors. They can be used throughout Rubix in Estimators that use the concept of distance to make predictions such as K Nearest Neighbors, K Means, and Local Outlier Factor.

Canberra
A weighted version of Manhattan distance which computes the L1 distance between two coordinates in a vector space.

Parameters:
This Kernel does not have any parameters.

Example:
use Rubix\ML\Kernels\Distance\Canberra;

$kernel = new Canberra();
Cosine
Cosine Similarity is a measure that ignores the magnitude of the distance between two vectors thus acting as strictly a judgement of orientation. Two vectors with the same orientation have a cosine similarity of 1, two vectors oriented at 90° relative to each other have a similarity of 0, and two vectors diametrically opposed have a similarity of -1. To be used as a distance function, we subtract the Cosine Similarity from 1 in order to satisfy the positive semi-definite condition, therefore the Cosine distance is a number between 0 and 2.

Parameters:
This Kernel does not have any parameters.

Example:
use Rubix\ML\Kernels\Distance\Cosine;

$kernel = new Cosine();
Diagonal
The Diagonal (sometimes called Chebyshev) distance is a measure that constrains movement to horizontal, vertical, and diagonal from a point. An example that uses Diagonal movement is a chess board.

Parameters:
This Kernel does not have any parameters.

Example:
use Rubix\ML\Kernels\Distance\Diagonal;

$kernel = new Diagonal();
Ellipsoidal
The Ellipsoidal distance measures the distance between two points on a 3-dimensional ellipsoid.

Parameters:
This Kernel does not have any parameters.

Example:
use Rubix\ML\Kernels\Distance\Ellipsoidal;

$kernel = new Ellipsoidal();
Euclidean
This is the ordinary straight line (bee line) distance between two points in Euclidean space. The associated norm of the Euclidean distance is called the L2 norm.

Parameters:
This Kernel does not have any parameters.

Example:
use Rubix\ML\Kernels\Distance\Euclidean;

$kernel = new Euclidean();
Hamming
The Hamming distance is defined as the sum of all coordinates that are not exactly the same. Therefore, two coordinate vectors a and b would have a Hamming distance of 2 if only one of the three coordinates were equal between the vectors.

Parameters:
This Kernel does not have any parameters.

Example:
use Rubix\ML\Kernels\Distance\Hamming;

$kernel = new Hamming();
Manhattan
A distance metric that constrains movement to horizontal and vertical, similar to navigating the city blocks of Manhattan. An example that used this type of movement is a checkers board.

Parameters:
This Kernel does not have any parameters.

Example:
use Rubix\ML\Kernels\Distance\Manhattan;

$kernel = new Manhattan();
Minkowski
The Minkowski distance is a metric in a normed vector space which can be considered as a generalization of both the Euclidean and Manhattan distances. When the lambda parameter is set to 1 or 2, the distance is equivalent to Manhattan and Euclidean respectively.

Parameters:
#	Param	Default	Type	Description
1	lambda	3.0	float	Controls the curvature of the unit circle drawn from a point at a fixed distance.
Example:
use Rubix\ML\Kernels\Distance\Minkowski;

$kernel = new Minkowski(4.0);
Cross Validation
Cross validation is the process of testing the generalization performance of a computer model using various techniques. Rubix has a number of classes that run cross validation on an instantiated Estimator for you. Each Validator outputs a scalar score based on the chosen metric.

Validators
Validators take an Estimator instance, Labeled Dataset object, and validation Metric and return a validation score that measures the generalization performance of the model using one of various cross validation techniques. There is no need to train the Estimator beforehand as the Validator will automatically train it on subsets of the dataset chosen by the testing algorithm.

public test(Estimator $estimator, Labeled $dataset, Validation $metric) : float
Example:
use Rubix\ML\CrossValidation\KFold;
use Rubix\ML\CrossValidation\Metrics\Accuracy;

...
$validator = new KFold(10);

$score = $validator->test($estimator, $dataset, new Accuracy());

var_dump($score);
Output:
float(0.869)
Below describes the various Cross Validators available in Rubix.

Hold Out
Hold Out is the simplest form of cross validation available in Rubix. It uses a hold out set equal to the size of the given ratio of the entire training set to test the model. The advantages of Hold Out is that it is quick, but it doesn’t allow the model to train on the entire training set.

Parameters:
#	Param	Default	Type	Description
1	ratio	0.2	float	The ratio of samples to hold out for testing.
Example:
use Rubix\ML\CrossValidation\HoldOut;
use Rubix\ML\CrossValidation\Metrics\Accuracy;

$validator = new HoldOut(0.25);
K Fold
K Fold is a technique that splits the training set into K individual sets and for each training round uses 1 of the folds to measure the validation performance of the model. The score is then averaged over K. For example, a K value of 10 will train and test 10 versions of the model using a different testing set each time.

Parameters:
#	Param	Default	Type	Description
1	k	10	int	The number of times to split the training set into equal sized folds.
Example:
use Rubix\ML\CrossValidation\KFold;

$validator = new KFold(5);
Leave P Out
Leave P Out tests the model with a unique holdout set of P samples for each round until all samples have been tested. Note that this process can become slow with large datasets and small values of P.

Parameters:
#	Param	Default	Type	Description
1	p	10	int	The number of samples to leave out each round for testing.
Example:
use Rubix\ML\CrossValidation\LeavePOut;

$validator = new LeavePOut(30);
Validation Metrics
Validation metrics are for evaluating the performance of an Estimator given some ground truth such as class labels. The output of the Metric’s score() method is a scalar score. You can output a tuple of minimum and maximum scores with the range() method.

To compute a validation score on an Estimator with a Labeled Dataset:

public score(Estimator $estimator, Labeled $testing) : float
To output the range of values the metric can take on:

public range() : array
Example:
use Rubix\ML\CrossValidation\Metrics\MeanAbsoluteError;

...
$metric = new MeanAbsoluteError();

$score = $metric->score($estimator, $testing);

var_dump($metric->range());

var_dump($score);
Output:
array(2) {
  [0]=> float(-INF)
  [1]=> int(0)
}

float(-0.99846070553066)
There are different metrics for the different types of Estimators listed below.

Anomaly Detection
Metric	Range	Description
Accuracy	(0, 1)	A quick metric that computes the accuracy of the detector.
F1 Score	(0, 1)	A metric that takes the precision and recall into consideration.
Classification
Metric	Range	Description
Accuracy	(0, 1)	A quick metric that computes the accuracy of the classifier.
F1 Score	(0, 1)	A metric that takes the precision and recall of into consideration.
Informedness	(0, 1)	Measures the probability of making an informed prediction by looking at the sensitivity and specificity of each class outcome.
MCC	(-1, 1)	Matthews Correlation Coefficient is a coefficient between the observed and predicted binary classifications. A coefficient of +1 represents a perfect prediction, 0 no better than random prediction, and −1 indicates total disagreement between prediction and label.
Clustering
Metric	Range	Description
Completeness	(0, 1)	A measure of the class outcomes that are predicted to be in the same cluster.
Concentration	(-INF, INF)	A score that measures the ratio between the within-cluster dispersion and the between-cluster dispersion (also called Calinski Harabaz score).
Homogeneity	(0, 1)	A measure of the cluster assignments that are known to be in the same class.
V Measure	(0, 1)	The harmonic mean between Homogeneity and Completeness.
Example:
use Rubix\ML\CrossValidation\Metrics\Accuracy;
use Rubix\ML\CrossValidation\Metrics\Homogeneity;

$metric = new Accuracy();

$metric = new Homogeneity();
Regression
Metric	Range	Description
Mean Absolute Error	(-INF, 0)	The average absolute difference between the actual and predicted values.
Median Absolute Error	(-INF, 0)	The median absolute difference between the actual and predicted values.
Mean Squared Error	(-INF, 0)	The average magnitude or squared difference between the actual and predicted values.
RMS Error	(-INF, 0)	The root mean squared difference between the actual and predicted values.
R-Squared	(-INF, 1)	The R-Squared value, or sometimes called coefficient of determination is the proportion of the variance in the dependent variable that is predictable from the independent variable(s).
Reports
Reports allow you to evaluate the performance of a model with a testing set. To generate a report, pass a trained Estimator and a testing Dataset to the Report’s generate() method. The output is an associative array that can be used to generate visualizations or other useful statistics.

To generate a report:

public generate(Estimator $estimator, Dataset $dataset) : array
Example:
use Rubix\ML\Classifiers\KNearestNeighbors;
use Rubix\ML\Reports\ConfusionMatrix;
use Rubix\ML\Datasets\Labeled;

...
$dataset = new Labeled($samples, $labels);

list($training, $testing) = $dataset->randomize()->split(0.8);

$estimator = new KNearestNeighbors(7);

$report = new ConfusionMatrix(['positive', 'negative', 'neutral']);

$estimator->train($training);

$result = $report->generate($estimator, $testing);
Most Reports require a Labeled dataset because they need some sort of ground truth to go compare to. The Reports that are available in Rubix are listed below.

Aggregate Report
A Report that aggregates the results of multiple reports. The reports are indexed by their order given at construction time.

Parameters:
#	Param	Default	Type	Description
1	reports	[ ]	array	An array of Report objects to aggregate.
Example:
use Rubix\ML\Reports\AggregateReport;
use Rubix\ML\Reports\ConfusionMatrix;
use Rubix\ML\Reports\MulticlassBreakdown;
use Rubix\ML\Reports\PredictionSpeed;

...
$report = new AggregateReport([
	new ConfusionMatrix(['wolf', 'lamb']),
	new MulticlassBreakdown(),
	new PredictionSpeed(),
]);

$result = $report->generate($estimator, $testing);
Confusion Matrix
A Confusion Matrix is a table that visualizes the true positives, false, positives, true negatives, and false negatives of a Classifier. The name stems from the fact that the matrix makes it easy to see the classes that the Classifier might be confusing.

Classification
Parameters:
#	Param	Default	Type	Description
1	classes	All	array	The classes to compare in the matrix.
Example:
use Rubix\ML\Reports\ConfusionMatrix;

...
$report = new ConfusionMatrix(['dog', 'cat', 'turtle']);

$result = $report->generate($estimator, $testing);

var_dump($result);
Output:
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
Contingency Table
A Contingency Table is used to display the frequency distribution of class labels among a clustering of samples.

Clustering
Parameters:
This Report does not have any parameters.

Example:
use Rubix\ML\Reports\ContingencyTable;

...
$report = new ContingencyTable();

$result = $report->generate($estimator, $testing);

var_dump($result);
Output:
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
Multiclass Breakdown
A Report that drills down in to each unique class outcome. The report includes metrics such as Accuracy, F1 Score, MCC, Precision, Recall, Cardinality, Miss Rate, and more.

Classification
Parameters:
This Report does not have any parameters.

Example:
use Rubix\ML\Reports\MulticlassBreakdown;

...
$report = new MulticlassBreakdown();

$result = $report->generate($estimator, $testing);

var_dump($result);
Output:
...
    array(2) {
      ["benign"]=>
      array(15) {
        ["accuracy"]=> int(1)
        ["precision"]=> float(0.99999999998723)
        ["recall"]=> float(0.99999999998723)
        ["specificity"]=> float(0.99999999998812)
        ["miss_rate"]=> float(1.2771450563775E-11)
        ["fall_out"]=> float(1.1876499783625E-11)
        ["f1_score"]=> float(0.99999999498723)
        ["mcc"]=> float(0.99999999999998)
        ["informedness"]=> float(0.99999999997535)
        ["true_positives"]=> int(783)
        ["true_negatives"]=> int(842)
        ["false_positives"]=> int(0)
        ["false_negatives"]=> int(0)
        ["cardinality"]=> int(783)
        ["density"]=> float(0.48184615384615)
      }
...
Outlier Ratio
Outlier Ratio is the proportion of outliers to inliers in an Anomaly Detection problem. It can be used to gauge the amount of contamination that the Detector is predicting.

Anomaly Detection
Parameters:
This Report does not have any parameters.

Example:
use Rubix\ML\Reports\OutlierRatio;

...
$report = new OutlierRatio();

$result = $report->generate($estimator, $testing);

var_dump($result);
Output:
  array(4) {
    ["outliers"]=> int(13)
    ["inliers"]=> int(307)
    ["ratio"]=> float(0.042345276871585)
    ["cardinality"]=> int(320)
  }
Prediction Speed
This Report measures the number of predictions an Estimator can make per second as given by the PPS (predictions per second) score.

Classification, Regression, Clustering, Anomaly Detection
Parameters:
This Report does not have any parameters.

Example:
use Rubix\ML\Reports\PredictionSpeed;

...
$report = new PredictionSpeed();

$result = $report->generate($estimator, $testing);

var_dump($result);
Output:
  array(4) {
    ["ppm"]=> float(4332968.1101788)
    ["average_seconds"]=> float(1.3847287706694E-5)
    ["total_seconds"]=> float(0.0041680335998535)
    ["cardinality"]=> int(301)
  }

Residual Analysis
Residual Analysis is a type of Report that measures the differences between the predicted and actual values of a regression problem.

Regression
Parameters:
This Report does not have any parameters.

Example:
use Rubix\ML\Reports\ResidualAnalysis;

...
$report = new ResidualAnalysis();

$result = $report->generate($estimator, $testing);

var_dump($result);
Output:
  array(9) {
    ["mean_absolute_error"]=>
    float(2.1971189157834)
    ["median_absolute_error"]=>
    float(1.714)
    ["mean_squared_error"]=>
    float(8.7020753279997)
    ["rms_error"]=>
    float(2.9499280208167)
    ["min"]=>
    float(0.0069999999999908)
    ["max"]=>
    float(14.943333333333)
    ["variance"]=>
    float(3.8747437979066)
    ["r_squared"]=>
    float(0.82286934000174)
    ["cardinality"]=>
    int(301)
  }
Other
This section includes broader functioning objects that aren’t part of a specific category.

Guessing Strategies
Guesses can be thought of as a type of weak prediction. Unlike a real prediction, guesses are made using limited information and basic means. A guessing Strategy attempts to use such information to formulate an educated guess. Guessing is utilized in both Dummy Estimators (Dummy Classifier, Dummy Regressor) as well as the Missing Data Imputer.

The Strategy interface provides an API similar to Transformers as far as fitting, however, instead of being fit to an entire dataset, each Strategy is fit to an array of either continuous or discrete values.

To fit a Strategy to an array of values:

public fit(array $values) : void
To make a guess based on the fitted data:

public guess() : mixed
Example:
use Rubix\ML\Other\Strategies\BlurryMedian;

$values = [1, 2, 3, 4, 5];

$strategy = new BlurryMedian(0.05);

$strategy->fit($values);

var_dump($strategy->range()); // Min and max guess for continuous strategies

var_dump($strategy->guess());
var_dump($strategy->guess());
var_dump($strategy->guess());
Output:
array(2) {
  [0]=> float(2.85)
  [1]=> float(3.15)
}

float(2.897176548)
float(3.115719462)
float(3.105983314)
Strategies are broken up into the Categorical type and the Continuous type. You can output the set of all possible categorical guesses by calling the set() method on any Categorical Strategy. Likewise, you can call range() on any Continuous Strategy to output the minimum and maximum values the guess can take on.

Here are the guessing Strategies available to use in Rubix.

Blurry Mean
This continuous Strategy that adds a blur factor to the mean of a set of values producing a random guess around the mean.

Continuous
Parameters:
#	Param	Default	Type	Description
1	blur	0.2	float	The amount of Gaussian noise by ratio of the standard deviation to add to the guess.
Example:
use Rubix\ML\Other\Strategies\BlurryMean;

$strategy = new BlurryMean(0.05);
Blurry Median
Adds random Gaussian noise to the median of a set of values.

Continuous
Parameters:
#	Param	Default	Type	Description
1	blur	0.2	float	The amount of Gaussian noise by ratio of the interquartile range to add to the guess.
Example:
use Rubix\ML\Other\Strategies\BlurryMedian;

$strategy = new BlurryMedian(0.5);
K Most Frequent
This Strategy outputs one of K most frequent discrete values at random.

Categorical
Parameters:
#	Param	Default	Type	Description
1	k	1	int	The number of most frequency categories to consider when formulating a guess.
Example:
use Rubix\ML\Other\Strategies\KMostFrequent;

$strategy = new KMostFrequent(5);
Lottery
Hold a lottery in which each category has an equal chance of being picked.

Categorical
Parameters:
This Strategy does not have any parameters.

Example:
use Rubix\ML\Other\Strategies\Lottery;

$strategy = new Lottery();
Popularity Contest
Hold a popularity contest where the probability of winning (being guessed) is based on the category’s prior probability.

Categorical
Parameters:
This Strategy does not have any parameters.

Example:
use Rubix\ML\Other\Strategies\Lottery;

$strategy = new PopularityContest();
Wild Guess
It’s just what you think it is. Make a guess somewhere in between the minimum and maximum values observed during fitting.

Continuous
Parameters:
This Strategy does not have any parameters.

Example:
use Rubix\ML\Other\Strategies\WildGuess;

$strategy = new WildGuess();
Tokenizers
Tokenizers take a body of text and converts it to an array of string tokens. Tokenizers are used by various algorithms in Rubix such as the Count Vectorizer to encode text into word counts.

To tokenize a body of text:

public tokenize(string $text) : array
Example:
use Rubix\ML\Other\Tokenizers\Word;

$text = 'I would like to die on Mars, just not on impact.';

$tokenizer = new Word();

var_dump($tokenizer->tokenize($text));
Output:
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
Below are the Tokenizers available in Rubix.

Whitespace
Tokens are delimited by a user-specified whitespace character.

Parameters:
#	Param	Default	Type	Description
1	delimiter	’ ’	string	The whitespace character that delimits each token.
Example:
use Rubix\ML\Other\Tokenizers\Whitespace;

$tokenizer = new Whitespace(',');
Word Tokenizer
Tokens are matched via regular expression designed to pick out words from a block of text. Note that this tokenizer will only pick up on words that are 2 or more characters.

Parameters:
This Tokenizer does not have any parameters.

Example:
use Rubix\ML\Other\Tokenizers\Word;

$tokenizer = new Word();
Testing
Rubix utilizes a combination of static analysis and unit tests to reduce errors in code. Rubix provides two Composer scripts that can be run from the root directory that automate the testing process.

To run static analysis:

composer analyze
To run the unit tests:

composer test
Contributing
Please make sure all your code is tested and passes static analysis (see Testing section above) before submitting it to the repository.

Markdown 121952 bytes 17796 words 3434 lines Ln 801, Col 48 HTML 89666 characters 15152 words 2971 paragraphs
