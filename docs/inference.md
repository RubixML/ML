# Inference
Inference is the process of making predictions using an estimator. You can think of an estimator as *inferring* the outcome of a sample given the input features and the estimator's hidden state obtained during training.

## Estimator Types
There are 4 base estimator types to consider in Rubix ML and each type outputs a prediction specific to its type. Meta-estimators can take on any one of these types depending on the base estimator that it wraps.

| Estimator Type | Output Prediction | Examples |
|---|---|---|
| Classifier | A categorical class label | cat, dog, ship |
| Regressor | A continuous value | 490,000 or 1.67592 |
| Clusterer | A discrete cluster number | 0, 1, 2, ..., n |
| Anomaly Detector | 1 for an anomaly, 0 otherwise | 0 or 1 |

## Making Predictions
All estimators implement the [Estimator](estimator.md) interface which provides the `predict()` method. The `predict()` method takes a dataset of unknown samples and returns their predictions from the model in an array. To return the predictions, pass the estimator a dataset containing unknown (unlabeled) samples with the same features that were used to train the model.

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
Sometimes, you'll just want to make a prediction on a single sample instead of an entire dataset. To return a single prediction from the model, pass the raw sample to the `predictSample()` method available on the [Learner](learner.md) interface ensuring that the features are given in the same order as when the learner was trained.

```php
$prediction = $estimator->predictSample([0.25, 3, 'furry']);

var_dump($prediction);
```

```sh
string(3) "cat"
```

## Estimation of Probabilities
Sometimes, you may want to know how *certain* the model is about a particular outcome. Classifiers and clusterers that implement the [Probabilistic](https://docs.rubixml.com/en/latest/probabilistic.html) interface have a `proba()` method that outputs the joint probability estimates for each class or cluster number as shown in the example below.

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
Certain anomaly detectors that implement the [Ranking](https://docs.rubixml.com/en/latest/ranking.html) interface can produce an anomaly score from the samples in a dataset. Anomaly scores are useful for attaining the degree of anomalousness for a sample. Higher anomaly scores equate to greater abnormality whereas low scores are typical of normal samples. Samples can be sorted by their anomaly score and the top samples can be flagged for further analysis.

```php
$scores = $estimator->rank($dataset);

var_dump($scores);
```

```sh
array(3) {
  [0]=> float(0.35033859096744)
  [1]=> float(0.40992076925443)
  [2]=> float(1.68163357834096)
}
```
