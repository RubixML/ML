# Inference
Inference is the process of making predictions using an estimator. Most estimators must be trained before they can make predictions because they implement the [Learner](learner.md) interface, but some estimators can make predictions by inferring information using only the current inputs. There are 4 types of estimators in Rubix ML and each type has a different output for a prediction. Estimators are organized into their own namespaces by type. Some meta-estimators are polymorphic i.e. they can become a classifier, or a regressor, or an anomaly detector based on the type of the estimator(s) they wrap. The [Estimator](estimator.md) interface provides the  `type()` method to return the integer-encoded estimator type of the instance.

### Estimator Outputs
| Estimator Type | Prediction | Examples |
|---|---|---|
| Classifier | A categorical class label | 'cat', 'dog', 'ship' |
| Regressor | A continuous value | 490,000 or 1.67592 |
| Clusterer | A discrete cluster number | '0', '1', '2', etc. |
| Anomaly Detector | '1' for an anomaly, '0' otherwise | '0' or '1' |
