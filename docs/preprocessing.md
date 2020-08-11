# Preprocessing
Sometimes, one or more preprocessing steps may need to be taken to transform the dataset before handing it off to a Learner. Some examples of preprocessing include feature extraction, standardization, normalization, imputation, and dimensionality reduction.

## Transformers
[Transformers](transformers/api.md) are objects that perform various preprocessing steps to the samples in a dataset. [Stateful](transformers/api.md#stateful) transformers are a type of transformer that must be *fitted* to a dataset. Fitting a dataset to a transformer is much like training a learner but in the context of preprocessing rather than inference. After fitting a stateful transformer, it will expect the features to be present in the same order when transforming subsequent datasets. A few transformers are *supervised* meaning they must be fitted with a [Labeled](datasets/labeled.md) dataset. [Elastic](transformers/api.md#elastic) transformers can have their fittings updated with new data after an initial fitting.

### Transform a Dataset
An example of a transformation is one that converts the categorical features of a dataset to continuous ones using a [*one hot*](https://en.wikipedia.org/wiki/One-hot) encoding. To accomplish this with the library, pass a [One Hot Encoder](transformers/one-hot-encoder.md) instance as an argument to the [Dataset](datasets/api.md) object's `apply()` method. Note that the `apply()` method also handles fitting automatically.

```php
use Rubix\ML\Transformers\OneHotEncoder;

$dataset->apply(new OneHotEncoder());
```

Transformations can be chained by calling the `apply()` method fluently.

```php
use Rubix\ML\Transformers\RandomHotDeckImputer;
use Rubix\ML\Transformers\OneHotEncoder;
use Rubix\ML\Transformers\ZScaleStandardizer;

$dataset->apply(new RandomHotDeckImputer(5))
    ->apply(new OneHotEncoder())
    ->apply(new ZScaleStandardizer());
```

> **Note:** Transformers do not alter the labels in a dataset. Instead, you can use the `transformLabels()` method on a [Labeled](https://docs.rubixml.com/en/latest/datasets/labeled.html#transform-labels) dataset instance.

### Manually Fitting
If for some reason you need to fit a stateful transformer to a dataset other than the one it was meant to transform, you can fit the transformer manually by calling the `fit()` method before applying the transformation.

```php
use Rubix\ML\Transformers\RandomHotDeckImputer;

$transformer = new RandomHotDeckImputer(5);

$transformer->fit($dataset1);

$dataset2->apply($transformer);
```

## Transform a Single Column
Sometimes, we might just want to transform a single column of the dataset. In the example below we use the `transformColumn()` method on the dataset to log transform a specified column.

```php
$dataset->transformColumn(6, 'log1p');
```

## Types of Preprocessing

### Standardization and Normalization
Oftentimes, the continuous features of a dataset will be on different scales because they were measured by different methods. For example, age (0 - 100) and income (0 - 9,999,999) are on two widely different scales. Standardization is the processes of transforming a dataset such that the features are all on one scale. Normalization is the special case where the transformed features have a range between 0 and 1. Depending on the transformer, it may operate on the columns or the rows of the dataset.

| Transformer | Operates On | Range | Stateful | Elastic |
|---|---|---|---|---|
| [L1 Normalizer](transformers/l1-normalizer.md) | Rows | [0, 1] | | |
| [L2 Normalizer](transformers/l2-normalizer.md) | Rows | [0, 1] | | |
| [Max Absolute Scaler](transformers/max-absolute-scaler.md) | Columns | [-1, 1] | ● | ● |
| [Min Max Normalizer](transformers/min-max-normalizer.md) | Columns | [min, max] | ● | ● |
| [Robust Standardizer](transformers/robust-standardizer.md) | Columns | [-∞, ∞] | ● | |
| [Z Scale Standardizer](transformers/z-scale-standardizer.md) | Columns | [-∞, ∞] | ● | ● |

### Feature Conversion
Feature converters are transformers that convert feature columns of one type to another. Since learners can be compatible with different data types, it may be necessary sometimes to convert features of an incompatible type to a compatible one.

| Transformer | From | To | Stateful | Elastic |
|---|---|---|---|---|
| [Interval Discretizer](transformers/interval-discretizer.md) | Continuous | Categorical | ● | |
| [One Hot Encoder](transformers/one-hot-encoder.md) | Categorical | Continuous | ● | |
| [Numeric String Converter](transformers/numeric-string-converter.md) | Categorical | Continuous | | |

### Dimensionality Reduction
Dimensionality reduction in machine learning is analogous to compression in the context of sending data over a wire. It allows a learner to train and infer quicker by producing a dataset with fewer but more informative features.

| Transformer | Supervised | Stateful | Elastic |
|---|---|---|---|
| [Dense Random Projector](transformers/dense-random-projector.md) | | ● | |
| [Gaussian Random Projector](transformers/gaussian-random-projector.md) | | ● | |
| [Linear Discriminant Analysis](transformers/linear-discriminant-analysis.md) | ● | ● | |
| [Principal Component Analysis](transformers/principal-component-analysis.md) | | ● | |
| [Sparse Random Projector](transformers/sparse-random-projector.md) | | ● | |

### Feature Selection
Similarly to dimensionality reduction, feature selection aims to reduce the number of features in a dataset, however, feature selection seeks to keep the best features as-is and drop the less informative ones entirely. Adding feature selection can help speed up training and inference by creating a more parsimonious model. It can also improve the performance of the model by removing *noise* features and features that are uncorrelated with the outcome.

| Transformer | Supervised | Stateful | Elastic |
|---|---|---|---|
| [Variance Threshold Filter](transformers/variance-threshold-filter.md) | | ● | |

### Imputation
One technique for handling missing data is a preprocessing step called *imputation*. Imputation is the process of replacing missing values in the dataset with a pretty good substitution. Examples include the average value for a feature or the sample's nearest neighbor's value. Imputation allows you to get more value from your data and can limit the introduction of bias in the process.

| Transformer | Continuous | Categorical | Stateful | Elastic |
|---|---|---|---|---|
| [KNN Imputer](transformers/knn-imputer.md) | ● | ● | ● | |
| [Missing Data Imputer](transformers/missing-data-imputer.md) | ● | ● | ● | |
| [Random Hot Deck Imputer](transformers/random-hot-deck-imputer.md) | ● | ● | ● | |

### Text Transformers
The library provides a number of transformers for natural language processing (NLP) tasks such as those for text cleaning, normalization, and feature extraction. Cleaning the text will help eliminate noise such as *stop words* or other uninformative tokens like URLs and email addresses from the corpus. Normalizing the text ensures that words like `therapist`, `Therapist`, and `ThErApIsT` are recognized as the same word. Feature extractors such as [Word Count Vectorizer](transformers/word-count-vectorizer.md) encode text features as fixed-length numerical feature vectors for input to a learner.

| Transformer | Stateful | Elastic |
|---|---|---|
| [HTML Stripper](transformers/html-stripper.md) | | |
| [Regex Filter](transformers/regex-filter.md) | | |
| [Text Normalizer](transformers/text-normalizer.md) | | |
| [Multibyte Text Normalizer](transformers/multibyte-text-normalizer.md) | | |
| [Stop Word Filter](transformers/stop-word-filter.md) | | |
| [TF-IDF Transformer](transformers/tf-idf-transformer.md) | ● | ● |
| [Whitespace Trimmer](transformers/whitespace-trimmer.md) | | |
| [Word Count Vectorizer](transformers/word-count-vectorizer.md) | ● | |

### Image Transformers
| Transformer | Stateful | Elastic |
|---|---|---|
| [Image Resizer](transformers/image-resizer.md) | | |
| [Image Vectorizer](transformers/image-vectorizer.md) | ● | |

### Other Transformers
| Transformer | Stateful | Elastic |
|---|---|---|
| [Polynomial Expander](transformers/polynomial-expander.md) | | |

## Transformer Pipelines
[Pipeline](pipeline.md) meta-estimators help you automate a series of transformations. In addition, Pipeline objects are [Persistable](persistable.md) allowing you to save and load transformer fittings between processes. Whenever a dataset object is passed to a learner wrapped in a Pipeline, it will automatically be fitted and/or transformed before it arrives in the learner's context.

Let's apply the same 3 transformers as in the example above by passing the transformer instances in the order we want them applied along with a base estimator to the constructor of Pipeline like in the example below.

```php
use Rubix\ML\Pipeline;
use Rubix\ML\Transformers\RandomHotDeckImputer;
use Rubix\ML\Transformers\OneHotEncoder;
use Rubix\ML\Transformers\ZScaleStandardizer;
use Rubix\ML\Classifiers\SoftmaxClassifier;

$estimator = new Pipeline([
    new RandomHotDeckImputer(5),
    new OneHotEncoder(),
    new ZScaleStandardizer(),
], new SoftmaxClassifier(200));
```

Calling `train()` or `partial()` will result in the transformers being fitted or updated before being passed to the Softmax Classifier.

```php
$estimator->train($dataset); // Transformers fitted and applied automatically

$estimator->partial($dataset); // Transformers updated and applied
```

Any time a dataset is passed to the Pipeline it will automatically be transformed before being handed to the underlying estimator.

```php
$predictions = $estimator->predict($dataset); // Dataset automatically transformed
```

## Saving a Dataset
If you ever want to preprocess a dataset and then save it for later you can do so by calling one of the conversion methods (`toCSV()`, `toNDJSON()`, etc.) on the [Dataset](datasets/api.md#encode-the-dataset) object to return an encoding that can be written directly to disk at a specified path.

```php
use Rubix\ML\Transformers\MissingDataImputer;

$dataset->apply(new MissingDataImputer())->toCSV()->write('dataset.csv');
```