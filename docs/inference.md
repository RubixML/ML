# Inference
Inference is the process of making predictions using an [Estimator](estimator.md). You can think of an estimator *inferring* the outcome of a sample given the input features and the estimator's hidden state obtained during training. Once a learner has been trained it can perform inference on any number of unknown samples containing values of the features used to train it.

## Estimator Types
There are 4 base estimator types to consider in Rubix ML and each type outputs a prediction specific to its type. Meta-estimators are *polymorphic* in the sense that they take on the type of the base estimator they wrap.

| Estimator Type | Prediction | PHP Type |
|---|---|---|
| Classifier | Class label | String |
| Regressor | Number | Integer or Floating Point Number |
| Clusterer | Discrete cluster number | Integer |
| Anomaly Detector | 1 for an anomaly, 0 otherwise | Integer |

## Making Predictions
All estimators implement the [Estimator](estimator.md) interface which provides the `predict()` method. The `predict()` method takes a dataset of unknown samples and returns their predictions from the model in an array.

> **Note:** The inference samples must contain the same number and order of feature columns as the samples used to train the learner.

```php
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
Sometimes, you'll just want to make a prediction on a single sample instead of an entire dataset. To return a single prediction from the model, pass the raw sample to the `predictSample()` method available on the [Learner](learner.md) interface.

```php
$prediction = $estimator->predictSample([0.25, 3, 'furry']);

var_dump($prediction);
```

```sh
string(3) "cat"
```

## Estimation of Probabilities
Sometimes, you may want to know how *certain* the model is about a particular outcome. Classifiers and clusterers that implement the [Probabilistic](https://docs.rubixml.com/en/latest/probabilistic.html) interface have the `proba()` method that computes the joint probability estimates for each class or cluster number as shown in the example below.

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

## Anomaly Scores
Anomaly detectors that implement the [Ranking](https://docs.rubixml.com/en/latest/ranking.html) interface can output the anomaly scores assigned to the samples in a dataset. Anomaly scores are useful for attaining the degree of anomalousness for a sample relative to other samples. Higher anomaly scores equate to greater abnormality whereas low scores are typical of normal samples.

```php
$scores = $estimator->score($dataset);

var_dump($scores);
```

```sh
array(3) {
  [0]=> float(0.35033859096744)
  [1]=> float(0.40992076925443)
  [2]=> float(1.68163357834096)
}
```
