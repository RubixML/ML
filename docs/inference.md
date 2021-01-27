# Inference
Inference is the process of making predictions using an [Estimator](estimator.md). You can think of an estimator *inferring* the outcome of a sample given the input features and the estimator's hidden state obtained during training. Once a learner has been trained it can perform inference on any number of samples.

!!! note
    As of version 0.3.0, single sample inference methods have been marked internal. As such, you should not rely on their API in your systems. Instead, use the associated dataset inference method with a dataset object containing a single sample.

## Estimator Types
There are 4 base estimator types to consider in Rubix ML and each type outputs a prediction specific to its type. Meta-estimators are *polymorphic* in the sense that they take on the type of the base estimator they wrap.

| Estimator Type | Prediction | Data Type | Example |
|---|---|---|---|
| Classifier | Class label | String | 'cat', 'positive' |
| Regressor | Number | Integer or Float | 42, 1.348957 |
| Clusterer | Cluster number | Integer | 0, 15 |
| Anomaly Detector | 1 for an anomaly or 0 otherwise | Integer | 0, 1 |

## Making Predictions
All estimators implement the [Estimator](estimator.md) interface which provides the `predict()` method. The `predict()` method takes a dataset of unknown samples and returns their predictions from the model in an array.

!!! note
    The inference samples must contain the same number and order of feature columns as the samples used to train the learner.

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

## Estimation of Probabilities
Sometimes, you may want to know how *certain* the model is about a particular outcome. Classifiers and clusterers that implement the [Probabilistic](probabilistic.md) interface have the `proba()` method that computes the joint probability estimates for each class or cluster number as shown in the example below.

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
Anomaly detectors that implement the [Scoring](scoring.md) interface can output the anomaly scores assigned to the samples in a dataset. Anomaly scores are useful for attaining the degree of anomalousness for a sample relative to other samples. Higher anomaly scores equate to greater abnormality whereas low scores are typical of normal samples.

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
