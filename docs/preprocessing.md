# Preprocessing
Sometimes, one or more preprocessing steps will need to be taken to condition your data for a learner. Some examples of preprocessing include feature extraction, standardization, normalization, imputation, and dimensionality reduction. Preprocessing in Rubix ML is handled through [Transformer](transformers/api.md) objects whose logic is hidden behind an easy-to-use interface. Each transformer performs a pass over the samples in a dataset and alters the features in some way. [Stateful](transformers/api.md#stateful) transformers need to be *fitted* with a training set before they can transform a dataset. [Elastic](transformers/api.md#elastic) transformers can have their fittings updated in much the same way an online learner can be partially trained.

A common transformation involves converting the categorical features of a dataset to continuous ones using a *one hot* encoding. To accomplish this with the library, pass a [One Hot Encoder](transformers/one-hot-encoder.md) instance to a [Dataset](datasets/api.md) object's `apply()` method which automatically takes care of fitting and transforming the samples.

**Example**

```php
use Rubix\ML\Transformers\OneHotEncoder;

$dataset->apply(new OneHotEncoder());
```

Transformations can be chained by calling the `apply()` method fluently.

**Example**

```php
use Rubix\ML\Transformers\NumericStringConverter;
use Rubix\ML\Transformers\RandomHotDeckImputer;
use Rubix\ML\Transformers\OneHotEncoder;

$dataset->apply(new NumericStringConverter())
    ->apply(new RandomHotDeckImputer(2))
    ->apply(new OneHotEncoder());
```

## Standardization and Normalization
Often, the continuous features of a dataset will be on different scales because they are measured differently. For example, age (0 - 100) and income (0 - 9,999,999) are on two widely different scales. Standardization is the processes of transforming a dataset such that the features are all on one scale. Normalization is the special case where the transformed features have a range between 0 and 1. Depending on the transformer, it may operate on the columns or the rows of the dataset.

| Transformer | Operates On | Stateful | Elastic |
|---|---|---|---|
| [L1 Normalizer](transformers/l1-normalizer.md) | Rows | | |
| [L2 Normalizer](transformers/l2-normalizer.md) | Rows | | |
| [Max Absolute Scaler](transformers/max-absolute-scaler.md) | Columns | ● | ● |
| [Min Max Normalizer](transformers/min-max-normalizer.md) | Columns | ● | ● |
| [Robust Standardizer](transformers/robust-standardizer.md) | Columns | ● | |
| [Z Scale Standardizer](transformers/z-scale-standardizer.md) | Columns | ● | ● |

## Feature Conversion
Sometimes we are stuck in a situation when we have a dataset with both categorical and continuous features but the learner is only compatible with one of those types. For this issue we'll need to convert the incompatible type to a compatible type in order to proceed to train the learner.

| Transformer | From | To | Stateful | Elastic |
|---|---|---|---|---|
| [Interval Discretizer](transformers/interval-discretizer.md) | Continuous | Categorical | ● | |
| [One Hot Encoder](transformers/one-hot-encoder.md) | Categorical | Continuous | ● | |
| [Numeric String Converter](transformers/numeric-string-converter.md) | Categorical | Continuous | | |

## Imputation
A technique for handling missing data is a preprocessing step called *imputation*. Imputation is the process of replacing missing values in the dataset with a pretty good substitution. Examples include the average value for a feature or the sample's nearest neighbor's value. Imputation allows you to get more value from your data and limits the introduction of certain biases in the process.

**Examples**

| Transformer | Categorical | Continuous | Stateful | Elastic |
|---|---|---|---|---|
| [KNN Imputer](transformers/knn-imputer.md) | ● | ● | ● | ● |
| [Missing Data Imputer](transformers/missing-data-imputer.md) | ● | ● | ● | |
| [Random Hot Deck Imputer](transformers/random-hot-deck-imputer.md) | ● | ● | ● | ● |

## Feature Extraction
Higher-order data such as images and text blobs are actually composites of many features. Thus, it is often necessary to extract those features from their original representation in order to feed them to a learner.

| Transformer | Source | Stateful | Elastic |
|---|---|---|---|
| [Image Vectorizer](transformers/image-vectorizer.md) | Images | | |
| [Word Count Vectorizer](transformers/word-count-vectorizer.md) | Text Blobs | ● | |

## Dimensionality Reduction
Dimensionality reduction in machine learning is analogous to compression in the context of sending data over a wire. It allows a learner to train and infer quicker by producing a dataset with fewer but more informative features.

**Examples**

- [Dense Random Projector](transformers/dense-random-projector.md)
- [Gaussian Random Projector](transformers/gaussian-random-projector.md)
- [Linear Discriminant Analysis](transformers/linear-discriminant-analysis.md)
- [Principal Component Analysis](transformers/principal-component-analysis.md)
- [Sparse Random Projector](transformers/sparse-random-projector.md)

## Feature Selection
Similarly to dimensionality reduction, feature selection aims to reduce the number of features in a dataset, however, feature selection seeks to keep the best features as-is and drop the less informative ones entirely. Adding feature selection can help speed up training and inference by creating a more parsimonious model. It can also improve the performance of the model by removing *noise* features and features that are uncorrelated with the outcome.

**Examples**

- [Variance Threshold Filter](transformers/variance-threshold-filter.md)

## Image Processing
For computer vision tasks, images may need to be processed to ensure they are the correct size and shape. Other forms of image processing may include color correction and blurring/sharpening.

**Example**

- [Image Resizer](transformers/image-resizer.md)

## Text Cleaning
For natural language processing (NLP) tasks, cleaning the text will help eliminate noise such as *stop words* or other uninformative tokens like URLs and email addresses from the corpus. Another common step is to *normalize* the text so that words like `therapist`, `Therapist`, and `ThErApIsT` are recognized as the same word.

**Examples**

- [HTML Stripper](transformers/html-stripper.md)
- [Regex Filter](transformers/regex-filter.md)
- [Text Normalizer](transformers/text-normalizer.md)
- [Stop Word Filter](transformers/stop-word-filter.md)

## Transformer Pipelines
[Pipeline](pipeline.md) meta-estimators help you automate a series of transformations. In addition, Pipeline objects are [Persistable](persistable.md) which allow you to save and load transformer fittings between processes. Whenever a dataset object is passed to a learner wrapped in a Pipeline, it will transparently be fitted and/or transformed in the background before it arrives in the method context.

Let's say we wanted to build a pipeline to normalize some blobs of text, extract the term frequencies (TF), and then transform them by their inverse document frequency (IDF). We could build such a transformer Pipeline by passing the transformer instances in the order we want them applied along with a base estimator to its constructor like in the example below.

**Example**

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

Calling `train()` or `partial()` will result in the transformers being fitted or updated before being passed to the underlying learner.

**Example**

```php
$estimator->train($dataset); // Transformers fitted and applied automatically

$estimator->partial($dataset); // Transformers updated and applied
```

Any time a dataset is passed to the Pipeline it will automatically be transformed before being handed to the underlying estimator.

**Example**

```php
$predictions = $estimator->predict($dataset); // Dataset automatically transformed
```