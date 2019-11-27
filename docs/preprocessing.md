# Preprocessing
Sometimes, one or more preprocessing steps will need to be taken to condition the incoming data for a learner. Some examples of various types of data preprocessing are feature extraction, standardization, normalization, imputation, and dimensionality reduction. Preprocessing in Rubix ML is handled through [Transformer](https://docs.rubixml.com/en/latest/transformers/api.html) objects whose logic is hidden behind an easy to use interface. Say we wanted to transform the categorical features of a dataset to continuous ones using *one-hot* encoding - we can accomplish this in Rubix ML by passing a [One Hot Encoder](https://docs.rubixml.com/en/latest/transformers/one-hot-encoder.html) instance as an argument to a dataset object's `apply()` method like in the example below.

**Example**

```php
use Rubix\ML\Transformers\OneHotEncoder;

$dataset->apply(new OneHotEncoder());
```

## Standardization and Normalization
Often, the continuous features of a dataset will be on different scales due to different forms of measurement. For example, age (0 - 100) and income (0 - 1,000,000) are on two vastly different scales. The condition that all features are on the same scale matters to some learners such as [K Nearest Neighbors](https://docs.rubixml.com/en/latest/classifiers/k-nearest-neighbors.html), [K Means](https://docs.rubixml.com/en/latest/clusterers/k-means.html), and [Multilayer Perceptron](https://docs.rubixml.com/en/latest/classifiers/multilayer-perceptron.html). Standardization is often accompanied by a *centering* step which gives the transformed feature matrix 0 mean. Depending on the transformer, it may operate on the columns of the feature matrix or the rows. Normalization is the special case where the transformed features have a range between 0 and 1.

### Column-wise Standardizers
- [Max Absolute Scaler](https://docs.rubixml.com/en/latest/transformers/max-absolute-scaler.html)
- [Min Max Normalizer](https://docs.rubixml.com/en/latest/transformers/min-max-normalizer.html)
- [Robust Standardizer](https://docs.rubixml.com/en/latest/transformers/robust-standardizer.html)
- [Z Scale Standardizer](https://docs.rubixml.com/en/latest/transformers/z-scale-standardizer.html)

### Row-wise Standardizers
- [L1 Normalizer](https://docs.rubixml.com/en/latest/transformers/l1-normalizer.html)
- [L2 Normalizer](https://docs.rubixml.com/en/latest/transformers/l2-normalizer.html)

## Imputation
Although some estimators are robust to missing data, the primary tool for handling missing data in Rubix ML is through a process called imputation. Data imputation is the process of replacing missing values with a pretty good substitution such as the average value or the sample's nearest neighbor's value. By imputing values rather than discarding or ignoring the sample we minimize the introduction of bias. The available imputers are listed below.

- [KNN Imputer](https://docs.rubixml.com/en/latest/transformers/knn-imputer.html)
- [Missing Data Imputer](https://docs.rubixml.com/en/latest/transformers/missing-data-imputer.html)
- [Random Hot Deck Imputer](https://docs.rubixml.com/en/latest/transformers/random-hot-deck-imputer.html)

## Feature Extraction

On the map ...

## Dimensionality Reduction

On the map ...

## Transformer Pipelines

On the map ...