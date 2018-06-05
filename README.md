# Rubix for PHP
Rubix is a library that lets you build intelligent programs that learn from data in PHP.

### Our Mission
The goal of the Rubix project is to bring state-of-the-art machine learning capabilities to the PHP language. Although the language is primarily optimized to deliver performance on the web, we believe this should *not* prevent PHP programmers from taking advantage of the major advances in AI and machine learning today. Our intent is to provide the tooling to facilitate small to medium sized projects, rapid prototyping, and education.

### Index

 - [Installation](#installation)
 - Introduction
	 - [Obtaining Data](#obtaining-data)
	 - [Choosing an Estimator](#choosing-an-estimator)
	 - [Training and Prediction](#training-and-prediction)
	 - [Evaluation](#evaluating-model-performance)
	 - [Next Steps](#what-next)
- API Reference
	- [Estimators](#estimators)
		- Classifiers
			- [AdaBoost](#adaboost)
			- [Decision Tree](#decision-tree)
			- [Dummy Classifier](#dummy-classifier)
			- [K Nearest Neighbors](#k-nearest-neighbors)
			- [Logistic Regression](#logistic-regression)
			- [Multi Layer Perceptron](#multi-layer-perceptron)
			- [Naive Bayes](#naive-bayes)
			- [Random Forest](#random-forest)
			- [Softmax Classifier](#softmax-classifier)
		- Regressors
			- [Dummy Regressor](#dummy-regressor)
			- [KNN Regressor](#knn-regressor)
			- [MLP Regressor](#mlp-regressor)
			- [Regression Tree](#regression-tree)
			- [Ridge](#ridge)
		- Clusterers
			- [DBSCAN](#dbscan)
			- [Fuzzy C Means](#fuzzy-c-means)
			- [K Means](#k-means)
	- Data Preprocessing
		- [Pipeline](#pipeline)
		- [Transformers](#transformers)
			- [L1 and L2 Regularizers](#l1-and-l2-regularizers)
			- [Min Max Normalizer](#min-max-normalizer)
			- [Missing Data Imputer](#missing-data-imputer)
			- [Numeric String Converter](#numeric-string-converter)
			- [One Hot Encoder](#one-hot-encoder)
			- [Stop Word Filter](#stop-word-filter)
			- [Text Normalizer](#text-normalizer)
			- [TF-IDF Transformer](#tf---idf-transformer)
			- [Token Count Vectorizer](#token-count-vectorizer)
			- [Variance Threshold Filter](#variance-threshold-filter)
			- [Z Scale Standardizer](#z-scale-standardizer)
	- Cross Validation
	- Model Selection
	- Model Persistence
- Requirements
- Licence

## Installation
Install Rubix using composer:
```sh
composer require rubix/engine
```

## An Introduction to Machine Learning in Rubix
Machine learning is the process by which a computer program is able to progressively improve performance on a certain task through training and data without explicitly being programmed. There are two types of learning techniques that Rubix offers out of the box, **Supervised** and **Unsupervised**.
 - **Supervised** learning is a technique to train computer models with a dataset in which the outcome of each sample data point has been *labeled* either by a human expert or another ML model prior to training. There are two types of supervised learning to consider in Rubix:
	 - **Classification** is the problem of identifying which *class* a particular sample belongs to among a set of categories. For example, one task may be in determining a particular species of Iris flower based on its sepal and petal dimensions.
	 - **Regression** involves predicting continuous *values* rather than discrete classes. An example in which a regression model is appropriate would be predicting the life expectancy of a population based on economic factors.
- **Unsupervised** learning, by contrast, uses an *unlabeled* dataset and instead relies on discovering information through the features of the training samples alone.
	- **Clustering** is the process of grouping data points in such a way that members of the same group are more similar (homogeneous) than the rest of the samples. You can think of clustering as assigning a class label to an otherwise unlabeled sample. An example where clustering might be used is in differentiating tissues in PET scan images.

### Obtaining Data
Machine learning projects typically begin with a question. For example, who of my friends are most likely to stay married to their spouse? One way to go about answering this question with machine learning would be to go out and ask a bunch of long-time married and divorced couples the same set of questions and then use that data to build a model of what a successful (or not) marriage looks like. Later, you can use that model to make predictions based on the answers from your friends.

Although this is certainly a valid way of obtaining data, in reality, chances are someone has already done the work of measuring the data for you and it is your job to find it, aggregate it, clean it, and otherwise make it usable by the machine learning algorithm. There are a number of PHP libraries out there that make extracting data from CSV, JSON, databases, and cloud services a whole lot easier, and we recommend checking them out before attempting it manually.

Having that said, Rubix will be able to handle any dataset as long as it can fit into one its predefined Dataset objects (Labeled, Unlabeled, etc.).

#### The Dataset Object
All of the machine learning algorithms (called *Estimators*) in Rubix require a Dataset object to train. Unlike standard PHP arrays, Dataset objects extend the basic data structure functionality with many useful features such as properly splitting, folding, and randomizing the data points.

For the following example, suppose that you went out and asked 100 couples (50 married and 50 divorced) to rate (between 1 and 5) their similarity, communication, and partner attractiveness. We can construct a Labeled Dataset object from the data you collected in the following way:

```php
use \Rubix\Engine\Datasets\Labeled;

$samples = [[3, 4, 2], [1, 5, 3], [4, 4, 3], [2, 1, 5], ...];

$labels = ['married', 'divorced', 'married', 'divorced', ...];

$dataset = new Labeled($samples, $labels);
```

### Choosing an Estimator

There are many different algorithms to chose from and each one is designed to handle specific (often overlapping) tasks. Choosing the right Estimator for the job is crucial to building an accurate and performant computer model.

There are a couple ways that we could model our marriage satisfaction predictor. We could have asked a fourth question - that is, to rate each couple's overall marriage satisfaction and then train a Regressor to predict a continuous "satisfaction score" for each new sample. But since all we have to go by for now is whether they are still married or currently divorced, a Classifier will be better suited.

In practice, one will experiment with more than one type of Classifier to find the best fit to the data, but for the purposes of this introduction we will simply demonstrate a common and intuitive algorithm called *K Nearest Neighbors*.

#### Creating the Estimator Instance

Like most Estimators, the K Nearest Neighbors Classifier requires a number of parameters (called *Hyperparameters*) to be chosen up front. These parameters can be chosen based on some prior knowledge of the problem space, or at random. Rubix provides a meta-Estimator called Grid Search that, given a list, searches the parameter space for the most effective combination. For the purposes of this example we will just go with our intuition and chose the parameters outright.

Here are the hyperparameters for K Nearest Neighbors:

| Param | Default | Type | Description |
|--|--|--|--|
| k | 5 | int | The number of neighboring training samples to consider when making a prediction. |
| distance | Euclidean | object | The distance metric used to measure the distance between two sample points. |

The K Nearest Neighbors algorithm works by comparing the "distance" between a given sample and each of the training samples. It will then use the K nearest samples to base its prediction on. For example, if the 5 closest neighbors to a sample are 4 married and 1 divorced, the algorithm will output a prediction of married with a probability of 0.80.

It is important to understand the effect that each parameter has on the performance of the particular Estimator as different values can often lead to drastically different results.

To create a K Nearest Neighbors Classifier instance:
```php
use Rubix\Engine\Classifiers\KNearestNeighbors;
use Rubix\Engine\Metrics\Distance\Manhattan;

// Using the default parameters
$estimator = new KNearestNeighbors();

// Specifying parameters
$estimator = new KNearestNeighbors(3, new Manhattan());
```
### Training and Prediction
Now that we've chosen and instantiated an Estimator and our Dataset object is ready to go, it is time to train our model and use it to make some predictions.

Using the Labeled Dataset object we created earlier we can train the KNN estimator like such:
```php
$estimator->train($dataset);
```
For our 100 sample dataset, this will only take a few milliseconds, but larger datasets and more sophisticated Estimators can take much longer.

Once our Estimator has been trained we can feed in some new sample points to see what the model predicts. Suppose that we went out and collected  new data points from our friends using the same questions we asked the married and divorced couples. We could make a prediction on whether they look more like the class of love birds or the class of divorcees by taking their answers and doing the following:
```php
use Rubix\Engine\Dataset\Unlabeled;

$samples = [[4, 1, 3], [2, 2, 1], [2, 4, 5], [5, 2, 4]];

$friends = new Unlabeled($samples);

$predictions = $estimator->predict($friends);

var_dump($predictions);
```
Outputs:
```sh
array(5) {
	[0] => 'divorced',
	[1] => 'divorced',
	[2] => 'married',
	[3] => 'married',
}
```
Note that we are not using a Labeled Dataset here because we don't know the outcomes yet. In fact, the outcome is exactly what we are trying to predict.

### Evaluating Model Performance
We can make predictions but how do we know that the predictions our model makes are accurate? The answer to that is called *cross validation* and Rubix has a number of tools that can automate the process of evaluating a model. For the purposes of this introduction, we will use the simplest form of cross validation called Holdout. The idea is to randomize and split the dataset into a training and testing set, such that a portion of the data is "held out" to be used to test the model.

We need to supply a validation metric to measure suited to our Classifier. In this case we will use the Accuracy metric, but many more exist depending on your purpose.
```php
use Rubix\Engine\CrossValidation\HoldOut;
use Rubix\Engine\Metrics\Validation\Accuracy;

$validator = new HoldOut(new Accuracy(), 0.2);

$score = $validator->score($estimator, $dataset);

var_dump($score);
```
Outputs:
```sh
float(0.945)
```
Since we are measuring accuracy, this output means that our Estimator is about 95% accurate given the data we've provided it. The second HoldOut parameter 0.2 instructs the validator to use 20% of the dataset for testing. More data for testing means the result will have less variance however that also means we don't use as much data to train the model.

### What Next?
Now that we've gone through a brief introduction of a simple machine learning problem in Rubix, the next step is to become more familiar with the API and to experiment with some data on your own. We highly recommend reading the entire documentation, but if you're eager to get started with Rubix and are comfortable with machine learning a great place to get started is with one of the many datasets available for free on the [University of California Irvine Machine Learning repository](https://archive.ics.uci.edu/ml/datasets.html) website.

## API Reference

### Estimators
Estimators are at the core of the Rubix library and consist of various Classifiers, Regressors, and Clusterers that make *predictions* based on the data they have been trained with. Estimators come in one of two types depending on how they are trained -  **Supervised** and **Unsupervised**.

The only difference to the API of either type of Estimator is the type of Dataset the model can be trained with.

##### Supervised:
```php
public train(Labeled $dataset) : void
```
##### Unsupervised:
```php
public train(Dataset $dataset) : void
```

Both types of Estimator share the same prediction API, however.

##### Supervised and Unsupervised:
```php
public predict(Dataset $samples) : array
```

The return array of a prediction is 0 indexed containing the predicted values of the supplied Dataset object in the same order.

##### Example:
```sh
array(3) {
	[0] => 'married',
	[1] => 'divorced',
	[2] => 'married',
}
```

Some Estimators may implement the **Probabilistic** interface, in which case, they will have an additional method that returns an array of probability scores for each class, cluster, etc. per sample in the given Dataset object.

```php
public proba(Dataset $samples) : array
```

##### Example:
```sh
array(3) {
	[0] => array(2) {
		['married'] => 0.975,
		['divorced'] => 0.025,
	}
	[1] => array(2) {
		['married'] => 0.200,
		['divorced'] => 0.800,
	}
	...
}
```
---
### Classifiers

#### AdaBoost
Short for Adaptive Boosting, this ensemble classifier can improve the performance of an otherwise weak classifier by focusing more attention on samples that are harder to classify.

##### Supervised, Binary Classifier, Persistable

##### Parameters:
| Param | Default | Type | Description |
|--|--|--|--|
| base | None | string | The fully qualified class name of the base "weak" classifier. |
| params | [ ] | array | The parameters of the base classifer. |
| epochs | 100 | int | The maximum number of training rounds to execute before the algorithm terminates. |
| ratio | 0.1 | float | The ratio of samples to subsample from the training dataset per epoch. |
| threshold | 0.999 | float | The minimum accuracy an epoch must score before the algorithm terminates. |

##### Additional Methods:
| Method | Description |
|--|--|
| `weights() : array` | Returns the calculated weight values of the last trained dataset. |
| `influence() : array` | Returns the influence scores for each boosted "weak" classifier.


##### Example:
```php
use Rubix\Engine\Classifiers\AdaBoost;
use Rubix\Engine\Classifiers\DecisionTree;

$estimator = new AdaBoost(DecisionTree::class, [1, 10], 100, 0.1, 0.999);

$estimator->weights(); // [0.25, 0.35, 0.1, ...]
$estimator->influence(); // [0.7522, 0.7945, ...]
```

#### Decision Tree
Binary Tree based algorithm that works by intelligently splitting the training data at various decision nodes until a terminating condition is met.

##### Supervised, Probabilistic, Persistable

##### Parameters:
| Param | Default | Type | Description |
|--|--|--|--|
| max depth | PHP_INT_MAX | int | The maximum depth of a branch that is allowed. Setting this to 1 is equivalent to training a Decision Stump. |
| min samples | 5 | int | The minimum number of data points needed to split a decision node. |

##### Additional Methods:
| Method | Description |
|--|--|
| `complexity() : int` | Returns the number of splits in the tree. |
| `height() : int` | Return the height of the tree. |
| `balance() : int` | Return the balance factor of the tree. |

##### Example:
```php
use Rubix\Engine\Classifiers\DecisionTree;

$estimator = new DecisionTree(10, 3);

$estimator->complexity(); // 20
$estimator->height(); // 9
$estimator->balance(); // -1
```

#### Dummy Classifier
A classifier based on a given imputer strategy. Used to compare performance with an actual classifier.

##### Supervised, Persistable

##### Parameters:
| Param | Default | Type | Description |
|--|--|--|--|
| strategy | PopularityContest | object | The imputer strategy to employ when guessing the outcome of a sample. |

##### Additional Methods:
This Estimator does not have any additional methods.|

##### Example:
```php
use Rubix\Engine\Classifiers\DummyClassifier;
use Rubix\Engine\Transformers\Strategies\PopularityContest;

$estimator = new DummyClassifier(new PopularityContest());
```

#### K Nearest Neighbors
A lazy learning algorithm that locates the K nearest samples from the training set and uses a majority vote to classify the unknown sample.

##### Supervised, Probabilistic

##### Parameters:
| Param | Default | Type | Description |
|--|--|--|--|
| k | 5 | int | The number of neighboring training samples to consider when making a prediction. |
| distance | Euclidean | object | The distance metric used to measure the distance between two sample points. |

##### Additional Methods:
This Estimator does not have any additional methods.|

##### Example:
```php
use Rubix\Engine\Classifiers\KNearestNeighbors;
use Rubix\Engine\Metrics\Distance\Euclidean;

$estimator = new KNearestNeighbors(3, new Euclidean());
```

#### Logistic Regression
A type of regression analysis that uses the logistic function to classify between two possible outcomes.

##### Supervised, Binary Classifier, Probabilistic, Persistable

##### Parameters:
| Param | Default | Type | Description |
|--|--|--|--|
| epochs | 100 | int | The number of training epochs to execute. |
| batch size | 10 | int | The number of training samples to process at a time. |
| optimizer | Adam | object | The gradient descent step optimizer used to train the underlying network. |
| alpha | 1e-4 | float | The L2 regularization term. |

##### Additional Methods:
This Estimator does not have any additional methods.|

##### Example:
```php
use Rubix\Engine\Classifers\LogisticRegression;
use Rubix\Engine\NeuralNet\Optimizers\Adam;

$estimator = new LogisticRegression(200, 10, new Adam(0.001), 1e-4);
```

#### Multi Layer Perceptron
Multiclass neural network model that uses a series of user-defined hidden layers as intermediate computational units equipped with non-linear activation functions.

##### Supervised, Probabilistic, Persistable

##### Parameters:
| Param | Default | Type | Description |
|--|--|--|--|
| hidden | [ ] | array | An array of hidden layers of the neural network. |
| batch size | 10 | int | The number of training samples to process at a time. |
| optimizer | Adam | object | The gradient descent step optimizer used to train the underlying network. |
| alpha | 1e-4 | float | The L2 regularization term. |
| metric | Accuracy | object | The validation metric used to monitor the training progress of the network. |
| ratio | 0.2 | float | The ratio of sample data to hold out for validation during training. |
| window | 3 | int | The number of epochs to consider when determining if the algorithm should terminate or keep training. |
| epochs | PHP_INT_MAX | int | The maximum number of training epochs to execute. |

##### Additional Methods:
| Method | Description |
|--|--|
| `progress() : array` | Returns an array with the validation score at each epoch of training. |

##### Example:
```php
use Rubix\Engine\Classifiers\MultiLayerPerceptron;
use Rubix\Engine\NeuralNet\Layers\Dense;
use Rubix\Engine\NeuralNet\ActivationFunctions\ELU;
use Rubix\Engine\NeuralNet\Optimizers\Adam;
use Rubix\Engine\Metrics\Validation\MCC;

$hidden = [
	new Dense(10, new ELU()),
	new Dense(10, new ELU()),
	new Dense(10, new ELU()),
];

$estimator = new MultiLayerPerceptron($hidden, 10, new Adam(0.001), 1e-4, new MCC(), 0.2, 3, PHP_INT_MAX);

$estimator->progress(); // [0.45, 0.59, 0.72, 0.88, ...]
```

#### Naive Bayes
Probability-based classifier that used probabilistic inference to derive the predicted class.

##### Supervised, Probabilistic, Persistable

##### Parameters:
This estimator does not have any hyperparameters.

##### Additional Methods:
This Estimator does not have any additional methods.|

##### Example:
```php
use Rubix\Engine\Classifiers\NaiveBayes;

$estimator = new NaiveBayes();
```

#### Random Forest
Ensemble classifier that trains Decision Trees on a random subset of the training data.

##### Supervised, Probabilistic, Persistable

##### Parameters:
| Param | Default | Type | Description |
|--|--|--|--|
| trees | 50 | int | The number of Decision Trees to train in the ensemble. |
| ratio | 0.1 | float | The ratio of random samples to train each Decision Tree with. |
| max depth | 10 | int | The maximum depth of a branch that is allowed. Setting this to 1 is equivalent to training a Decision Stump. |
| min samples | 5 | int | The minimum number of data points needed to split a decision node. |

##### Additional Methods:
This Estimator does not have any additional methods.|

##### Example:
```php
use Rubix\Engine\Classifiers\RandomForest;

$estimator = new RandomForest(100, 0.2, 5, 3);
```

#### Softmax Classifier
A generalization of logistic regression to multiple classes.

##### Supervised, Probabilistic, Persistable

##### Parameters:
| Param | Default | Type | Description |
|--|--|--|--|
| epochs | 100 | int | The number of training epochs to execute. |
| batch size | 10 | int | The number of training samples to process at a time. |
| optimizer | Adam | object | The gradient descent step optimizer used to train the underlying network. |
| alpha | 1e-4 | float | The L2 regularization term. |

##### Additional Methods:
This Estimator does not have any additional methods.|

##### Example:
```php
use Rubix\Engine\Classifiers\SoftmaxClassifier;
use Rubix\Engine\NeuralNet\Optimizers\Momentum;

$estimator = new SoftmaxClassifier(200, 10, new Momentum(0.001), 1e-4);
```

### Regressors
---

#### Dummy Regressor
Regressor that guesses the output values based on an imputer strategy. Used to compare performance against actual regressors.

##### Supervised, Persistable

##### Parameters:
| Param | Default | Type | Description |
|--|--|--|--|
| strategy | BlurryMean | object | The imputer strategy to employ when guessing the outcome of a sample. |

##### Additional Methods:
This Estimator does not have any additional methods.|

##### Example:
```php
use Rubix\Engine\Regressors\DummyRegressor;
use Rubix\Engine\Tranformers\Strategies\BlurryMean;

$estimator = new DummyRegressor(new BlurryMean());
```

#### KNN Regressor
A version of K Nearest Neighbors that uses the mean of K nearest data points to make a prediction.

##### Supervised

##### Parameters:
| Param | Default | Type | Description |
|--|--|--|--|
| k | 5 | int | The number of neighboring training samples to consider when making a prediction. |
| distance | Euclidean | object | The distance metric used to measure the distance between two sample points. |

##### Additional Methods:
This Estimator does not have any additional methods.|

##### Example:
```php
use Rubix\Engine\Regressors\KNNRegressor;
use Rubix\Engine\Metrics\Distance\Minkowski;

$estimator = new KNNRegressor(2, new Minkowski(3.0));
```

#### MLP Regressor
A neural network with a continuous output layer suitable for regression problems.

##### Supervised, Persistable

##### Parameters:
| Param | Default | Type | Description |
|--|--|--|--|
| hidden | [ ] | array | An array of hidden layers of the neural network. |
| batch size | 10 | int | The number of training samples to process at a time. |
| optimizer | Adam | object | The gradient descent step optimizer used to train the underlying network. |
| alpha | 1e-4 | float | The L2 regularization term. |
| metric | Accuracy | object | The validation metric used to monitor the training progress of the network. |
| ratio | 0.2 | float | The ratio of sample data to hold out for validation during training. |
| window | 3 | int | The number of epochs to consider when determining if the algorithm should terminate or keep training. |
| epochs | PHP_INT_MAX | int | The maximum number of training epochs to execute. |

##### Additional Methods:
| Method | Description |
|--|--|
| `progress() : array` | Returns an array with the validation score at each epoch of training. |

##### Example:
```php
use Rubix\Engine\Regressors\MLPRegressor;
use Rubix\Engine\NeuralNet\Layers\Dense;
use Rubix\Engine\NeuralNet\ActivationFunctions\HyperbolicTangent;
use Rubix\Engine\NeuralNet\ActivationFunctions\PReLU;
use Rubix\Engine\NeuralNet\Optimizers\RMSProp;
use Rubix\Engine\Metrics\Validation\RSquared;

$hidden = [
	new Dense(30, new HyperbolicTangent()),
	new Dense(50, new PReLU()),
];

$estimator = new MLPRegressor($hidden, 10, new RMSProp(0.001), 1e-2, new RSquared(), 0.2, 3, PHP_INT_MAX);

$estimator->progress(); // [0.66, 0.88, 0.94, 0.95, ...]
```

#### Regression Tree
A binary tree learning algorithm that minimizes the variance between decision node splits.

##### Supervised, Persistable

##### Parameters:
| Param | Default | Type | Description |
|--|--|--|--|
| max depth | PHP_INT_MAX | int | The maximum depth of a branch that is allowed. Setting this to 1 is equivalent to training a Decision Stump. |
| min samples | 5 | int | The minimum number of data points needed to split a decision node. |

##### Additional Methods:
| Method | Description |
|--|--|
| `complexity() : int` | Returns the number of splits in the tree. |
| `height() : int` | Return the height of the tree. |
| `balance() : int` | Return the balance factor of the tree. |

##### Example:
```php
use Rubix\Engine\Regressors\RegressionTree;

$estimator = new RegressionTree(50, 1);

$estimator->complexity(); // 36
$estimator->height(); // 18
$estimator->balance(); // 0
```

#### Ridge
L2 penalized least squares regression whose predictions are based on the closed-form solution to the training data.

##### Supervised, Persistable

##### Parameters:
| Param | Default | Type | Description |
|--|--|--|--|
| alpha | 1.0 | float | The L2 regularization term. |

##### Additional Methods:
| Method | Description |
|--|--|
| `intercept() : float` | Returns the y intercept of the computed regression line. |
| `coefficients() : array` | Return an array containing the computed coefficients of the regression line. |

##### Example:
```php
use Rubix\Engine\Regressors\Ridge;

$estimator = new Ridge(2.0);

$estimator->intercept(); // 5.298226
$estimator->coefficients(); // [2.023, 3.122, 5.401, ...]
```

### Clusterers
---

#### DBSCAN
Density-based spatial clustering of applications with noise is a clustering algorithm able to find non-linearly separable and arbitrarily-shaped clusters.

##### Unsupervised, Persistable

##### Parameters:
| Param | Default | Type | Description |
|--|--|--|--|
| radius | 0.5 | float | The maximum radius between two points for them to be considered in the same cluster. |
| min density | 5 | int | The minimum number of points within radius of each other to form a cluster. |
| distance | Euclidean | object | The distance metric used to measure the distance between two sample points.

##### Additional Methods:
This Estimator does not have any additional methods.|

##### Example:
```php
use Rubix\Engine\Clusterers\DBSCAN;
use Rubix\Engine\Metrics\Distance\Manhattan;

$estimator = new DBSCAN(4.0, 5, new Manhattan());
```

#### Fuzzy C Means
Clusterer that allows data points to belong to multiple clusters if they fall within a fuzzy region.

##### Unsupervised, Probabilistic, Persistable

##### Parameters:
| Param | Default | Type | Description |
|--|--|--|--|
| c | None | int | The number of target clusters. |
| fuzz | 2.0 | float | Determines the bandwidth of the fuzzy area. |
| distance | Euclidean | object | The distance metric used to measure the distance between two sample points. |
| threshold | 1e-4 | float | The minimum change in centroid means necessary for the algorithm to continue training. |
| epochs | PHP_INT_MAX | int | The maximum number of training rounds to execute. |

##### Additional Methods:
| Method | Description |
|--|--|
| `centroids() : array` | Returns an array of the C computed centroids of the training data. |
| `progress() : array` | Returns the progress of each epoch as the total distance between each sample and centroid. |

##### Example:
```php
use Rubix\Engine\Clusterers\FuzzyCMeans;
use Rubix\Engine\Metrics\Distance\Euclidean;

$estimator = new FuzzyCMeans(5, 2.5, new Euclidean(), 1e-3, 1000);

$estimator->centroids(); // [[3.149, 2.615], [-1.592, -3.444], ...]
$estimator->progress(); // [5878.01, 5200.50, 4960.28, ...]
```

#### K Means
A fast centroid-based hard clustering algorithm capable of clustering linearly separable data points.

##### Unsupervised, Persistable

##### Parameters:
| Param | Default | Type | Description |
|--|--|--|--|
| k | None | int | The number of target clusters. |
| distance | Euclidean | object | The distance metric used to measure the distance between two sample points. |
| epochs | PHP_INT_MAX | int | The maximum number of training rounds to execute. |

##### Additional Methods:
| Method | Description |
|--|--|
| `centroids() : array` | Returns an array of the K computed centroids of the training data. |

##### Example:
```php
use Rubix\Engine\Clusterers\KMeans;
use Rubix\Engine\Metrics\Distance\Euclidean;

$estimator = new KMeans(3, new Euclidean());

$estimator->centroids(); // [[3.149, 2.615], [-1.592, -3.444], ...]
```
---
### Data Preprocessing
Often, additional processing of input data is required to deliver correct predictions and/or accelerate the training process. In this section, we'll introduce the **Pipeline** meta-Estimator and the various **Transformers** that it employs to fit the input data to suit the requirements and preferences of the Estimator that it feeds.

#### Pipeline
A Pipeline is an Estimator that wraps another Estimator with additional functionality. For this reason, a Pipeline is called a *meta-Estimator*. As arguments, a Pipeline accepts a base Estimator and a list of Transformers to apply to the input data before it is fed to the learning algorithm. Under the hood, the Pipeline will automatically fit the training set upon training and transform any Dataset object supplied as an argument to one of the base Estimator's methods, including `predict()`.

##### Example:
```php
use Rubix\Engine\Pipeline;
use Rubix\Engine\Classifiers\SoftmaxClassifier;
use Rubix\Engine\Transformers\MissingDataImputer;
use Rubix\Engine\Transformers\OneHotEncoder;

$estimator = new Pipeline(new SoftmaxClassifier(200, 50), [
	new MissingDataImputer(),
	new OneHotEncoder(),
]);

$estimator->train($dataset); // Dataset objects are fit and
$estimator->predict($samples); // transformed automatically.
```

Transformer middleware will process in the order given when the Pipeline was built and cannot be reordered without instantiating a new one. Since Tranformers are run sequentially, the order in which they run *matters*. For example, a Transformer near the end of the stack may depend on a previous Transformer to convert all categorical features into continuous ones before it can run.

In practice, applying transformations can drastically improve the performance of your model by cleaning, scaling, expanding, compressing, and normalizing the input data. Below is a list of the available Transformers in Rubix.

### Transformers
Transformers are generally designed to be used by the Pipeline meta-Estimator, however they can be used manually as well.

The fit method will allow the transformer to compute any necessary information from the training set in order to carry out its transformations. You can think of *fitting* a Transformer like *training* an Estimator. Not all Transformers need to be fit to the training set, when in doubt do it anyways, it won't hurt.
```php
public fit(Dataset $dataset) : void
```

The transform method on the Transformer is not meant to be called directly. This is done to force transformations to be done on the Dataset object in place so to deal with large datasets. To transform a dataset you can simply pass the instantiated transformer to a Dataset object's `transform()` method.

##### Example:
```php
use Rubix\Engine\Transformers\MinMaxNormalizer;

$transformer = new MinMaxNormalizer();

$dataset->transform($transformer);
```

#### L1 and L2 Regularizers
Regularization terms are used to augment the input vector of each sample such that each feature is divided over the L1 or L2 norm (or "magnitude") of the vector.

##### Continuous *Only*
##### Parameters:
This Transformer does not have any parameters.

##### Additional Methods:
This Transformer does not have any additional methods.

##### Example:
```php
use Rubix\Engine\Transformers\L1Regularizer;
use Rubix\Engine\Transformers\L2Regularizer;

$transformer = new L1Regularizer();
$transformer = new L2Regularizer();
```

#### Min Max Normalizer
Min Max Normalization scales the input features from a range of 0 to 1 by dividing the feature value over the maximum value for that feature column.

##### Continuous
##### Parameters:
This Transformer does not have any parameters.

##### Additional Methods:
This Transformer does not have any additional methods.

##### Example:
```php
use Rubix\Engine\Transformers\MinMaxNormalizer;

$transformer = new MinMaxNormalizer();
```

#### Missing Data Imputer
In the real world, it is common to have to deal with datasets that are missing a few values here and there. The Missing Data Imputer searches for any missing value placeholders and replaces it with a guess based on a given imputer **Strategy**.

##### Categorical or Continuous
##### Parameters:
| Param | Default | Type | Description |
|--|--|--|--|
| placeholder | '?' | string or numeric | The placeholder that denotes a missing value. |
| continuous strategy | BlurryMean | object | The imputer strategy to employ for continuous feature columns. |
| categorical strategy | PopularityContest | object | The imputer strategy to employ for categorical feature columns. |

##### Additional Methods:
This Transformer does not have any additional methods.

##### Example:
```php
use Rubix\Engine\Transformers\MissingDataImputer;
use Rubix\Engine\Transformers\Strategies\BlurryMean;
use Rubix\Engine\Transformers\Strategies\PopularityContest;

$transformer = new MissingDataImputer('?', new BlurryMean(0.2), new PopularityContest());
```

#### Numeric String Converter
This handy little Transformer will convert all numeric strings into their integer or float counterparts. Useful for when extracting from a source that only recognizes data as string types.

##### Categorical
##### Parameters:
This Transformer does not have any parameters.

##### Additional Methods:
This Transformer does not have any additional methods.

##### Example:
```php
use Rubix\Engine\Transformers\NumericStringConverter;

$transformer = new NumericStringConverter();
```

#### One Hot Encoder
The One Hot Encoder takes a column of categorical features, such as the country a person was born in, and produces a one-hot vector of n-dimensions where n is equal to the number of unique categories of the feature column.

##### Categorical
##### Parameters:
| Param | Default | Type | Description |
|--|--|--|--|
| columns | Null | array | The user-specified columns to encode indicated by numeric index starting at 0. |

##### Additional Methods:
This Transformer does not have any additional methods.

##### Example:
```php
use Rubix\Engine\Transformers\OneHotEncoder;

$transformer = new OneHotEncoder([0, 3, 5, 7, 9]);
```

#### Stop Word Filter
For certain natural language processing (NLP) tasks it can be advantageous to remove common or ambiguous words from the dataset before being fed to the Estimator. The Stop Word Filter lets you do just that.

##### Categorical
##### Parameters:
| Param | Default | Type | Description |
|--|--|--|--|
| stop words | None | array | An array containing the words in their exact form to filter from the dataset. |

##### Additional Methods:
This Transformer does not have any additional methods.

##### Example:
```php
use Rubix\Engine\Transformers\StopWordFilter;

$transformer = new StopWordFilter(['bad', 'words', ...]);
```

#### Text Normalizer
This Transformer will convert all strings to lowercase and remove excess whitespace.

##### Categorical
##### Parameters:
| Param | Default | Type | Description |
|--|--|--|--|
| lowercase | True | boolean | Should the text be converted to all lowercase? |

##### Additional Methods:
This Transformer does not have any additional methods.

##### Example:
```php
use Rubix\Engine\Transformers\TextNormalizer;

$transformer = new TextNormalizer(true);
```

#### TF-IDF Transformer
Term Frequency - Inverse Document Frequency is the measure of how important a word is to a document. The TF-IDF value increases proportionally with the number of times a word appears in a document and is offset by the frequency of the word in the corpus. This Transformer makes the assumption that the input is made up of word frequency vectors such as those created by the Token Count Vectorizer.

##### Continuous *Only*
##### Parameters:
This Transformer does not have any parameters.

##### Additional Methods:
This Transformer does not have any additional methods.

##### Example:
```php
$transformer = new TfIdfTransformer();
```

#### Token Count Vectorizer
Word counts are often used to represent natural language as numerical vectors. The Token Count Vectorizer builds a vocabulary from the entire training set and then transforms every sample text column into a vector consisting of one column per vocabulary word having the value of the number of times that word appears in the column text.

##### Categorical
##### Parameters:
| Param | Default | Type | Description |
|--|--|--|--|
| max vocabulary | PHP_INT_MAX | int | The maximum number of words to encode into each word vector. |
| tokenizer | Word | object | The method of turning samples of text into individual tokens. |

##### Additional Methods:
| Method | Description |
|--|--|
| `vocabulary() : array` | Returns the vocabulary array. |
| `size() : int` | Returns the size of the vocabulary in number of tokens.

##### Example:
```php
use Rubix\Engine\Transformers\TokenCountVectorizer;
use Rubix\Engine\Transformers\Tokenizers\Word;

$transformer = new TokenCountVectorizer(5000, new Word());

$transformer->vocabulary(); // ['i', 'would', 'like', 'to', 'die', 'on', 'mars', 'just', 'not', 'on', 'impact', ...]
$transformer->size(); // 4722
```

#### Variance Threshold Filter
A type of feature selector that removes all columns that have a lower variance than the threshold. Variance is computed as the population variance of all the values in the feature column.

##### Categorical and Continuous
##### Parameters:
| Param | Default | Type | Description |
|--|--|--|--|
| threshold | 0.0 | float | The threshold at which lower scoring columns will be dropped from the dataset. |

##### Additional Methods:
| Method | Description |
|--|--|
| `selected() : array` | An array containing the columns that were selected during fitting. |

##### Example:
```php
use Rubix\Engine\Transformers\VarianceThresholdFilter;

$transformer = new VarianceThresholdFilter(500);
```

#### Z Scale Standardizer
A way of centering and scaling an input vector such that the values range from 0 to 1 with a unit variance.

##### Continuous
##### Parameters:
This Transformer does not have any parameters.

##### Additional Methods:
| Method | Description |
|--|--|
| `means() : array` | Return the means calculated by fitting the training set. |
| `stddevs() : array` | Return the standard deviations calculated during fitting. |

##### Example:
```php
use Rubix\Engine\Transformers\ZScaleStandardizer;

$transformer = new ZScaleStandardizer();
```
---
### Cross Validation
Documentation in the works ...

---
### Model Selection
Documentation in the works ...

---
### Model Persistence
Documentation in the works ...

## Requirements
- PHP 7.1.3 or above

## License
MIT
