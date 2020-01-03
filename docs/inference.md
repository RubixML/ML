# Inference
Most of a model's lifetime will be spent performing inference for analysis or production. Inference is the process of making predictions using an estimator. You can think of an estimator as *inferring* the outcome of a sample given the input features and the estimator's hidden state created during training.

## Making Predictions
All estimators implement the [Estimator](estimator.md) interface which provides the `predict()` method. The `predict()` method takes a dataset of samples as its only argument and returns their predictions in order. There are 4 base estimator types in Rubix ML and each type makes a prediction specific to its type.

| Estimator Type | Prediction | Examples |
|---|---|---|
| Classifier | A categorical class label | `cat`, `dog`, `ship` |
| Regressor | A continuous value | `490,000` or `1.67592` |
| Clusterer | A discrete cluster number | `0`, `1`, `2`, etc. |
| Anomaly Detector | `1` for an anomaly, `0` otherwise | `0` or `1` |

**Example**

```php
// Import dataset of unknown samples

$predictions = $estimator->predict($dataset);

var_dump($predictions);
```

```sh
array(3) {
  [0]=>
  string(3) "cat"
  [1]=>
  string(3) "dog"
  [2]=>
  string(4) "frog"
}
```

## Single Predictions
To make a prediction on a single sample, pass the raw sample to the `predictSample()` method available on the [Learner](learner.md) interface.

```php
$prediction = $estimator->predictSample([0.25, 3, 'furry']);

var_dump($prediction);
```

```sh
string(3) "cat"
```

## Estimation of Probabilities
Sometimes, you may also want to know how *certain* the model is about a particular outcome. Classifiers and clusterers that implement the [Probabilistic](https://docs.rubixml.com/en/latest/probabilistic.html) interface have a `proba()` method that outputs the joint probability estimates for each class or cluster number as shown in the example below.

**Example**
```php
$probabilities = $estimator->proba($dataset);  

var_dump($probabilities);
```

```sh
array(2) {
	[0] => array(2) {
		['monster'] => 0.975,
		['not monster'] => 0.025,
	}
	[1] => array(2) {
		['monster'] => 0.2,
		['not monster'] => 0.8,
	}
}
```

## Ranking Samples
Certain anomaly detectors that implement the [Ranking](https://docs.rubixml.com/en/latest/ranking.html) interface can output an anomaly score for each sample that can be used to sort the samples. Ranking is useful for identifying the top or bottom scoring samples. For example, you may want to set the top *k* anomalous samples aside for further analysis by a human expert.

**Example**

```php
$scores = $estimator->rank($dataset);

var_dump($scores);
```

```sh
array(3) {
  [0]=> float(0.35033859096744)
  [1]=> float(0.40992076925443)
  [2]=> float(0.68163357834096)
}
```