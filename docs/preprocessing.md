# Preprocessing
Sometimes, one or more preprocessing steps will need to be taken to condition the incoming data for a learner. Some examples of various types of data preprocessing are feature extraction, standardization, normalization, imputation, and dimensionality reduction. Preprocessing in Rubix ML is handled through [Transformer](transformers/api.md) objects whose logic is hidden behind an easy to use interface. Say we wanted to transform the categorical features of a dataset to continuous ones using *one-hot* encoding - we can accomplish this in Rubix ML by passing a [One Hot Encoder](transformers/one-hot-encoder.md) instance as an argument to a dataset object's `apply()` method like in the example below.

**Example**

```php
use Rubix\ML\Transformers\OneHotEncoder;

$dataset->apply(new OneHotEncoder());
```

## Standardization and Normalization
Often, the continuous features of a dataset will be on different scales due to different forms of measurement. For example, age (0 - 100) and income (0 - 1,000,000) are on two vastly different scales. The condition that all features are on the same scale matters to some learners such as [K Nearest Neighbors](classifiers/k-nearest-neighbors.md), [K Means](clusterers/k-means.md), and [Multilayer Perceptron](classifiers/multilayer-perceptron.md) to name a few. Standardization is often accompanied by a *centering* step which gives the transformed feature matrix 0 mean. Depending on the transformer, it may operate on the columns of the feature matrix or the rows. Normalization is a special case where the transformed features have a range between 0 and 1.

### Column-wise Standardizers
- [Max Absolute Scaler](transformers/max-absolute-scaler.md)
- [Min Max Normalizer](transformers/min-max-normalizer.md)
- [Robust Standardizer](transformers/robust-standardizer.md)
- [Z Scale Standardizer](transformers/z-scale-standardizer.md)

### Row-wise Standardizers
- [L1 Normalizer](transformers/l1-normalizer.md)
- [L2 Normalizer](transformers/l2-normalizer.md)

## Imputation
Although some estimators are robust to missing data, the primary tool for handling missing data in Rubix ML is through a process called imputation. Data imputation is the process of replacing missing values with a pretty good substitution such as the average value or the sample's nearest neighbor's value. By imputing values rather than discarding or ignoring the sample we minimize the introduction of bias. The available imputers are listed below.

- [KNN Imputer](transformers/knn-imputer.md)
- [Missing Data Imputer](transformers/missing-data-imputer.md)
- [Random Hot Deck Imputer](transformers/random-hot-deck-imputer.md)

## Feature Extraction
Certain forms of data such as text blobs and images do not have directly analogous feature representations. Thus, it is necessary to extract features from their original representation. For example, in order to extract a useful feature representation from a blob of text, the text must be encoded as some fixed-length feature vector. One way we can accomplish this is by computing a fixed-length *vocabulary* from the training corpus and then encoding each sample as a vector of word counts. This is exactly what the Word Count Vectorizer does under the hood.

- [Image Vectorizer](transformers/image-vectorizer.md)
- [Word Count Vectorizer](transformers/word-count-vectorizer.md)

## Dimensionality Reduction
According to the [Johnson-Lindenstrauss lemma](https://en.wikipedia.org/wiki/Johnson%E2%80%93Lindenstrauss_lemma), for every sample in high dimensions, there exists some lower dimensional embedding that nearly preserves the distances between points in Euclidean space. Therefore, it is often a practice to transform datasets in a way that results in denser features that train and infer quicker relative to their high-dimensional representation. Dimensionality reduction in machine learning is analogous to compressing a data stream before sending it over a wire.

- [Dense Random Projector](transformers/dense-random-projector.md)
- [Gaussian Random Projector](transformers/gaussian-random-projector.md)
- [Linear Discriminant Analysis](transformers/linear-discriminant-analysis.md)
- [Principal Component Analysis](transformers/principal-component-analysis.md)
- [Sparse Random Projector](transformers/sparse-random-projector.md)

## Transformer Pipelines

On the map ...