# Inference
Inference is the process of making predictions using an [Estimator](estimator.md). You can think of an estimator *inferring* the outcome of a sample given the input features and the estimator's hidden state obtained during training. Once a learner has been trained it can perform inference on any number of samples.

## Estimator Types
There are 4 base estimator types to consider in Rubix ML and each type outputs a prediction specific to its type. Meta-estimators are *polymorphic* in the sense that they take on the type of the base estimator they wrap.

| Estimator Type | Prediction | Data Type | Example |
|---|---|---|---|
| Classifier | Class label | String | 'cat' |
| Regressor | Number | Integer or Float | 1.348957 |
| Clusterer | Cluster number | Integer | 6 |
| Anomaly Detector | 1 for an anomaly or 0 otherwise | Integer | 0 |

## Making Predictions
All estimators implement the [Estimator](estimator.md) interface which provides the `predict()` method. The `predict()` method takes a dataset of unknown samples and returns their predictions from the model in an array.

!!! note
    The inference samples must contain the same number and order of feature columns as the samples used to train the learner.

```php
$predictions = $estimator->predict($dataset);

print_r($predictions);
```

```php
Array
(
    [0] => cat
    [1] => dog
    [2] => frog
)
```

## Estimation of Probabilities
Sometimes, you may want to know how *certain* the model is about a particular outcome. Classifiers and clusterers that implement the [Probabilistic](probabilistic.md) interface have the `proba()` method that computes the joint probability estimates for each class or cluster number as shown in the example below.

```php
$probabilities = $estimator->proba($dataset);  

print_r($probabilities);
```

```php
Array
(
    [0] => Array
        (
            [cat] => 0.6
            [dog] => 0.4
            [frog] => 0.0
        )
    [1] => Array
        (
            [cat] => 0.3
            [dog] => 0.6
            [frog] => 0.1
        )
    [2] => Array
        (
            [cat] => 0.0
            [dog] => 0.0
            [frog] => 1.0
        )
)
```

## Anomaly Scores
Anomaly detectors that implement the [Scoring](scoring.md) interface can output the anomaly scores assigned to the samples in a dataset. Anomaly scores are useful for attaining the degree of anomalousness for a sample relative to other samples. Higher anomaly scores equate to greater abnormality whereas low scores are typical of normal samples.

```php
$scores = $estimator->score($dataset);

print_r($scores);
```

```php
Array
(
    [0] => 0.35033
    [1] => 0.40992
    [2] => 1.68153
)
```
