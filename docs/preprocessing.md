# Preprocessing
Sometimes, one or more preprocessing steps will need to be taken to condition the incoming data for a learner. Some examples of various types of data preprocessing are feature extraction, standardization, normalization, imputation, and dimensionality reduction. Preprocessing in Rubix ML is handled through [Transformer](transformers/api.md) objects whose logic is hidden behind an easy to use interface. Say we wanted to transform the categorical features of a dataset to continuous ones using *one-hot* encoding - we can accomplish this in Rubix ML by passing a [One Hot Encoder](transformers/one-hot-encoder.md) instance as an argument to a [Dataset](datasets/api.md) object's `apply()` method like in the example below.

```php
use Rubix\ML\Transformers\OneHotEncoder;

$dataset->apply(new OneHotEncoder());
```

You can chain transformations by calling the Dataset object API fluently.

```php
use Rubix\ML\Transformers\NumericStringConverter;
use Rubix\ML\Transformers\RandomHotDeckImputer;
use Rubix\ML\Transformers\OneHotEncoder;

$dataset->apply(new NumericStringConverter())
    ->apply(new RandomHotDeckImputer(2))
    ->apply(new OneHotEncoder());
```

## Standardization and Normalization
Often, the continuous features of a dataset will be on different scales due to different forms of measurement. For example, age (0 - 100) and income (0 - 1,000,000) are on two vastly different scales. The condition that all features are on the same scale matters to some learners such as [K Nearest Neighbors](classifiers/k-nearest-neighbors.md), [K Means](clusterers/k-means.md), and [Multilayer Perceptron](classifiers/multilayer-perceptron.md) to name a few. Standardization is a transformation applied to the features of a dataset such that they are all on the same scale. Normalization is a special case where the transformed features have a range between 0 and 1. Standardization is often accompanied by a *centering* step which subtracts the mean. Depending on the transformer, it may either operate on the columns of a sample matrix or the rows.

**Column-wise Examples**

- [Max Absolute Scaler](transformers/max-absolute-scaler.md)
- [Min Max Normalizer](transformers/min-max-normalizer.md)
- [Robust Standardizer](transformers/robust-standardizer.md)
- [Z Scale Standardizer](transformers/z-scale-standardizer.md)

**Row-wise Examples**

- [L1 Normalizer](transformers/l1-normalizer.md)
- [L2 Normalizer](transformers/l2-normalizer.md)

## Feature Conversion
Sometimes we are stuck in a situation when we have a dataset with both categorical and continuous features but the learner is only compatible with one of those types. For this issue we'll need to convert the incompatible type to a compatible type in order to proceed. Rubix ML provides a number of transformers that convert between types automatically.

**Examples**

- [Interval Discretizer](https://docs.rubixml.com/en/latest/transformers/interval-discretizer.html)
- [One Hot Encoder](https://docs.rubixml.com/en/latest/transformers/one-hot-encoder.html)
- [Numeric String Converter](https://docs.rubixml.com/en/latest/transformers/numeric-string-converter.html)

## Imputation
Although some learners are robust to missing data, the primary tool for handling missing data in Rubix ML is through a preprocessing step called *imputation*. Data imputation is the process of replacing missing values with a pretty good substitution such as the average value for the column or the sample's nearest neighbor's value. By imputing values rather than discarding or ignoring them, we are able to squeeze more value from the data and limit the introduction of bias in the process.

**Examples**

- [KNN Imputer](transformers/knn-imputer.md)
- [Missing Data Imputer](transformers/missing-data-imputer.md)
- [Random Hot Deck Imputer](transformers/random-hot-deck-imputer.md)

## Feature Extraction
Certain forms of data such as text blobs and images do not have directly analogous scalar feature representations. Thus, it is necessary to extract features from their original representation. For example, to extract a useful feature representation from a blob of text, the text must be encoded as some fixed-length feature vector. One way we can accomplish this in Rubix ML is by computing a fixed-length *vocabulary* from the training corpus and then encode each sample as a vector of word (or *token*) counts. This is exactly what the Word Count Vectorizer does under the hood.

**Examples**

- [Image Vectorizer](transformers/image-vectorizer.md)
- [Word Count Vectorizer](transformers/word-count-vectorizer.md)

## Dimensionality Reduction
Dimensionality reduction in machine learning is analogous to compressing a data stream before sending it over a wire. According to the [Johnson-Lindenstrauss lemma](https://en.wikipedia.org/wiki/Johnson%E2%80%93Lindenstrauss_lemma), for every sample in high dimensions, there exists some lower-dimensional embedding that nearly preserves the distances between points in Euclidean space. In other words, datasets can almost always be represented with fewer but more informative features. Therefore, it is often a practice to transform a dataset in a way that results in denser features that train and infer quicker relative to their high-dimensional counterparts.

**Examples**

- [Dense Random Projector](transformers/dense-random-projector.md)
- [Gaussian Random Projector](transformers/gaussian-random-projector.md)
- [Linear Discriminant Analysis](transformers/linear-discriminant-analysis.md)
- [Principal Component Analysis](transformers/principal-component-analysis.md)
- [Sparse Random Projector](transformers/sparse-random-projector.md)

## Feature Selection
Similarly to dimensionality reduction, feature selection aims to reduce the number of features in a dataset - however, they work in different ways. Whereas dimensionality reduction produces denser representations using the information contained within *all* the features, feature selection seeks to keep the best features as-is and drop the less informative ones entirely. Adding feature selection as a preprocessing step can help speed up training and inference by creating a more parsimonious model. It can also improve the performance of the model by removing *noise* features and features that are uncorrelated with the outcome.

**Examples**

- [Variance Threshold Filter](https://docs.rubixml.com/en/latest/transformers/variance-threshold-filter.html)

## Transformer Pipelines
You can automate the application of a series of transformations to a dataset using the [Pipeline](pipeline.md) meta-estimator. Whenever a dataset is passed to an estimator wrapped in a Pipeline it will automatically be transformed before it hits the method context. Pipeline objects are also [Persistable](persistable.md) which allows you to save and load the state of the transformer fittings between processes. Let's say we wanted to build a pipeline to normalize some blobs of text, extract the word count vectors, and then transform them by their inverse document frequency - a common series of data transformations for natural language processing (NLP). We could build such a pipeline by passing the transformers in the order we want them applied along with a base estimator to Pipeline's constructor.

```php
use Rubix\ML\Pipeline;
use Rubix\ML\Transformers\TextNormalizer;
use Rubix\ML\Transformers\WordCountVectorizer;
use Rubix\ML\Transformers\TfIdfTransformer;
use Rubix\ML\Classifiers\GaussianNB;

$estimator = new Pipeline([
    new TextNormalizer(),
    new WordCountVectorizer(10000),
    new TfIdfTransformer(),
], new GaussianNB());
```