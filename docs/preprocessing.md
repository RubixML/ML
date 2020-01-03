# Preprocessing
Sometimes, one or more preprocessing steps will need to be taken to condition your data for a learner. Some examples of preprocessing include feature extraction, standardization, normalization, imputation, and dimensionality reduction. Preprocessing in Rubix ML is handled through [Transformer](transformers/api.md) objects whose logic is hidden behind an easy to use interface. They each perform a specific task that potentially involves altering the entire dataset in a single pass.

Say we wanted to transform the categorical features of a dataset to continuous ones using *one-hot* encoding - we can accomplish this in Rubix ML by passing a [One Hot Encoder](transformers/one-hot-encoder.md) instance as an argument to a [Dataset](datasets/api.md) object's `apply()` method like in the example below.

```php
use Rubix\ML\Transformers\OneHotEncoder;

$dataset->apply(new OneHotEncoder());
```

You can chain transformations by calling the dataset object API fluently.

```php
use Rubix\ML\Transformers\NumericStringConverter;
use Rubix\ML\Transformers\RandomHotDeckImputer;
use Rubix\ML\Transformers\OneHotEncoder;

$dataset->apply(new NumericStringConverter())
    ->apply(new RandomHotDeckImputer(2))
    ->apply(new OneHotEncoder());
```

## Standardization and Normalization
Often, the continuous features of a dataset will be on different scales because they are measured independently. For example, age (0 - 100) and income (0 - 9,999,999) are on two widely different scales. Standardization is the processes of transforming a dataset such that the features are all on one scale. Normalization is the special case where the transformed features have a range between 0 and 1. Depending on the transformer, it may operate on the columns of the dataset or on the rows.

| Transformer | Operates On |
|---|---|
| [L1 Normalizer](transformers/l1-normalizer.md) | Rows |
| [L2 Normalizer](transformers/l2-normalizer.md) | Rows |
| [Max Absolute Scaler](transformers/max-absolute-scaler.md) | Columns |
| [Min Max Normalizer](transformers/min-max-normalizer.md) | Columns |
| [Robust Standardizer](transformers/robust-standardizer.md) | Columns |
| [Z Scale Standardizer](transformers/z-scale-standardizer.md) | Columns |

## Feature Conversion
Sometimes we are stuck in a situation when we have a dataset with both categorical and continuous features but the learner is only compatible with one of those types. For this issue we'll need to convert the incompatible type to a compatible type in order to proceed to train the learner. Rubix ML provides a number of transformers that convert between types automatically.

| Transformer | From | To |
|---|---|---|
| [Interval Discretizer](transformers/interval-discretizer.md) | Continuous | Categorical |
| [One Hot Encoder](transformers/one-hot-encoder.md) | Categorical | Continuous |
| [Numeric String Converter](transformers/numeric-string-converter.md) | Categorical | Continuous |

## Imputation
A common technique for handling missing data is through a preprocessing step called *imputation*. Imputation is the process of replacing missing values with a pretty good substitution such as the average value for the feature or the sample's nearest neighbor's value. Imputing missing values, rather than ignoring them or discarding the entire sample, allows you to get the most from your data and limits the introduction of certain biases in the process.

**Examples**

- [KNN Imputer](transformers/knn-imputer.md)
- [Missing Data Imputer](transformers/missing-data-imputer.md)
- [Random Hot Deck Imputer](transformers/random-hot-deck-imputer.md)

## Dimensionality Reduction
Dimensionality reduction in machine learning is analogous to *compression* in the context of sending data over a wire. It allows a learner to train and infer quicker by producing a dataset with fewer but more informative features.

**Examples**

- [Dense Random Projector](transformers/dense-random-projector.md)
- [Gaussian Random Projector](transformers/gaussian-random-projector.md)
- [Linear Discriminant Analysis](transformers/linear-discriminant-analysis.md)
- [Principal Component Analysis](transformers/principal-component-analysis.md)
- [Sparse Random Projector](transformers/sparse-random-projector.md)

## Feature Extraction
Higher-order data such as images and text blobs are actually composites of many scalar features. Thus, it is often necessary to extract those features from their original representation in order to feed them to a learner. For example, we may want to extract color channel (RGB) data from an image or word counts from a blob of text.

| Transformer | Extracts From |
|---|---|
| [Image Vectorizer](transformers/image-vectorizer.md) | Images |
| [Word Count Vectorizer](transformers/word-count-vectorizer.md) | Text Blobs |

## Feature Selection
Similarly to dimensionality reduction, feature selection aims to reduce the number of features in a dataset, however, feature selection seeks to keep the best features as-is and drop the less informative ones entirely. Adding feature selection as a preprocessing step can help speed up training and inference by creating a more parsimonious model. It can also improve the performance of the model by removing *noise* features and features that are uncorrelated with the outcome.

**Examples**

- [Variance Threshold Filter](transformers/variance-threshold-filter.md)

## Image Processing
For computer vision tasks, images may need to be processed to ensure they are the correct size. Other forms of image processing may include color correction and blurring/sharpening.

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
You can automate the application of a series of transformations using the [Pipeline](pipeline.md) meta-estimator. In addition, Pipeline objects are [Persistable](persistable.md) which allow you to save and load the transformer fittings between processes.

Let's say we wanted to build a pipeline to normalize some blobs of text, extract the word count vectors, and then transform them by their inverse document frequency - a common series of transformations for natural language processing (NLP). We could build such a pipeline by passing the transformers in the order we want them applied along with a base estimator to Pipeline's constructor like in the example below.

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

When a dataset is passed to a method on an estimator wrapped in a Pipeline, it will automatically be transformed. Calling `train()` or `partial()` will result in the transformers being fitted and updated respectively before being passed to the underlying learner.

```php
$estimator->train($dataset); // Transformers fitted and applied automatically

$estimator->partial($dataset); // Transformers updated and applied

// ...

$predictions = $estimator->predict($dataset); // Dataset automatically transformed

$bar = $estimator->foo($dataset); // Dataset also transformed
```
