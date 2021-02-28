# Preprocessing
Sometimes, one or more preprocessing steps may need to be taken to transform the dataset before handing it off to a Learner. Some examples of preprocessing include feature extraction, standardization, normalization, imputation, and dimensionality reduction.

## Transformers
[Transformers](transformers/api.md) are objects that perform various preprocessing steps to the samples in a dataset. [Stateful](transformers/api.md#stateful) transformers are a type of transformer that must be *fitted* to a dataset. Fitting a dataset to a transformer is much like training a learner but in the context of preprocessing rather than inference. After fitting a stateful transformer, it will expect the features to be present in the same order when transforming subsequent datasets. A few transformers are *supervised* meaning they must be fitted with a [Labeled](datasets/labeled.md) dataset. [Elastic](transformers/api.md#elastic) transformers can have their fittings updated with new data after an initial fitting.

### Transform a Dataset
An example of a transformation is one that converts the categorical features of a dataset to continuous ones using a [*one hot*](https://en.wikipedia.org/wiki/One-hot) encoding. To accomplish this with the library, pass a [One Hot Encoder](transformers/one-hot-encoder.md) instance as an argument to the [Dataset](datasets/api.md) object's `apply()` method. Note that the `apply()` method also handles fitting a Stateful transformer automatically.

```php
use Rubix\ML\Transformers\OneHotEncoder;

$dataset->apply(new OneHotEncoder());
```

Transformations can be chained by calling the `apply()` method fluently.

```php
use Rubix\ML\Transformers\RandomHotDeckImputer;
use Rubix\ML\Transformers\OneHotEncoder;
use Rubix\ML\Transformers\MinMaxNormalizer;

$dataset->apply(new RandomHotDeckImputer(5))
    ->apply(new OneHotEncoder())
    ->apply(new MinMaxNormalizer());
```

!!! note
    Transformers do not alter the labels in a dataset. Instead, you can use the `transformLabels()` method on a [Labeled](datasets/labeled.md#transform-labels) dataset instance.

### Manually Fitting
If you need to fit a [Stateful](transformers/api.md#stateful) transformer to a dataset other than the one it was meant to transform, you can fit the transformer manually by calling the `fit()` method before applying the transformation.

```php
use Rubix\ML\Transformers\WordCountVectorizer;

$transformer = new WordCountVectorizer(5000);

$transformer->fit($dataset1);

$dataset2->apply($transformer);
```

### Update Fitting
To update the fitting of an [Elastic](transformers/api.md#elastic) transformer call the `update()` method with a new dataset.

```php
$transformer->update($dataset);
```

## Transform a Single Column
Sometimes, we just want to transform a single column of the dataset. In the example below, we use the `transformColumn()` method on the dataset object to perform a log transformation to a specified column offset by passing it the `log1p()` callback function to apply to each value in the column.

```php
$dataset->transformColumn(6, 'log1p');
```

In the next example, we'll convert the `null` values of another column to a special placeholder class `?`.

```php
$dataset->transformColumn(9, function ($value) {
    return $value === null ? '?' : $value;
});
```

## Standardization and Normalization
Oftentimes, the continuous features of a dataset will be on different scales because they were measured by different methods. For example, age (0 - 100) and income (0 - 9,999,999) are on two widely different scales. Standardization is the processes of transforming a dataset such that the features are all on one common scale. Normalization is the special case where the transformed features have a range between 0 and 1. Depending on the transformer, it may operate on the columns or the rows of the dataset.

| Transformer | Operates | Output Range | [Stateful](transformers/api.md#stateful) | [Elastic](transformers/api.md#elastic) |
|---|---|---|---|---|
| [L1 Normalizer](transformers/l1-normalizer.md) | Row-wise | [0, 1] | | |
| [L2 Normalizer](transformers/l2-normalizer.md) | Row-wise | [0, 1] | | |
| [Max Absolute Scaler](transformers/max-absolute-scaler.md) | Column-wise | [-1, 1] | ● | ● |
| [Min Max Normalizer](transformers/min-max-normalizer.md) | Column-wise | [min, max] | ● | ● |
| [Robust Standardizer](transformers/robust-standardizer.md) | Column-wise | [-∞, ∞] | ● | |
| [Z Scale Standardizer](transformers/z-scale-standardizer.md) | Column-wise | [-∞, ∞] | ● | ● |

## Feature Conversion
Feature converters are transformers that convert feature columns of one data type to another by changing their representation.

| Transformer | From | To | [Stateful](transformers/api.md#stateful) | [Elastic](transformers/api.md#elastic) |
|---|---|---|---|---|
| [Interval Discretizer](transformers/interval-discretizer.md) | Continuous | Categorical | ● | |
| [One Hot Encoder](transformers/one-hot-encoder.md) | Categorical | Continuous | ● | |
| [Numeric String Converter](transformers/numeric-string-converter.md) | Categorical | Continuous | | |
| [Boolean Converter](transformers/boolean-converter.md) | Other | Categorical or Continuous | | |

## Dimensionality Reduction
Dimensionality reduction is a preprocessing technique for projecting a dataset onto a lower dimensional vector space. It allows a learner to train and infer quicker by producing a training set with fewer but more informative features. Dimensionality reducers can also be used to visualize datasets by outputting low (1 - 3) dimensionality embeddings for use in plotting software.

| Transformer | Supervised | [Stateful](transformers/api.md#stateful) | [Elastic](transformers/api.md#elastic) |
|---|---|---|---|
| [Gaussian Random Projector](transformers/gaussian-random-projector.md) | | ● | |
| [Linear Discriminant Analysis](transformers/linear-discriminant-analysis.md) | ● | ● | |
| [Principal Component Analysis](transformers/principal-component-analysis.md) | | ● | |
| [Sparse Random Projector](transformers/sparse-random-projector.md) | | ● | |
| [Truncated SVD](transformers/truncated-svd.md) | | ● | |
| [t-SNE](embedders/t-sne.md) | | | |

## Feature Selection
Similarly to dimensionality reduction, feature selection aims to reduce the number of features in a dataset, however, feature selection seeks to keep the best features as-is and drop the less informative ones entirely. Adding feature selection can help speed up training and inference by creating a more parsimonious model. It can also improve the performance of the model by removing *noise* features and features that are uncorrelated with the outcome.

| Transformer | Supervised | [Stateful](transformers/api.md#stateful) | [Elastic](transformers/api.md#elastic) |
|---|---|---|---|
| [K Best Feature Selector](transformers/k-best-feature-selector.md) | ● | ● | |
| [Recursive Feature Eliminator](transformers/recursive-feature-eliminator.md) | ● | ● | |

## Imputation
A technique for handling missing values in your dataset is a preprocessing step called *imputation*. Imputation is the process of replacing missing values with a pretty good guess.

| Transformer | Data Compatibility | [Stateful](transformers/api.md#stateful) | [Elastic](transformers/api.md#elastic) |
|---|---|---|---|
| [KNN Imputer](transformers/knn-imputer.md) | Depends on distance kernel | ● | |
| [Missing Data Imputer](transformers/missing-data-imputer.md) | Categorical, Continuous | ● | |
| [Random Hot Deck Imputer](transformers/random-hot-deck-imputer.md) | Depends on distance kernel | ● | |

## Text Transformers
The library provides a number of transformers for natural language processing (NLP) and information retrieval (IR) tasks such as those for text cleaning, normalization, and feature extraction from raw text blobs.

| Transformer | [Stateful](transformers/api.md#stateful) | [Elastic](transformers/api.md#elastic) |
|---|---|---|
| [HTML Stripper](transformers/html-stripper.md) | | |
| [Regex Filter](transformers/regex-filter.md) | | |
| [Text Normalizer](transformers/text-normalizer.md) | | |
| [Multibyte Text Normalizer](transformers/multibyte-text-normalizer.md) | | |
| [Stop Word Filter](transformers/stop-word-filter.md) | | |
| [TF-IDF Transformer](transformers/tf-idf-transformer.md) | ● | ● |
| [Whitespace Trimmer](transformers/whitespace-trimmer.md) | | |
| [Word Count Vectorizer](transformers/word-count-vectorizer.md) | ● | |

## Image Transformers
These transformers operate on the high-level image data type.

| Transformer | [Stateful](transformers/api.md#stateful) | [Elastic](transformers/api.md#elastic) |
|---|---|---|
| [Image Resizer](transformers/image-resizer.md) | | |
| [Image Vectorizer](transformers/image-vectorizer.md) | ● | |

### Persisting Transformers
The persistence subsystem can be used to save and load any Stateful transformer that implements the [Persistable](persistable.md) interface. In the example below we'll fit a transformer to a dataset and then save it to the [Filesystem](persisters/filesystem.md) so we can load it in another process.

```php
use Rubix\ML\Persisters\Filesystem;

$transformer->fit($dataset);

$persister = new Filesystem('example.transformer');

$persister->save($transformer);
```

Then, to load the transformer in another process call the `load()` method on the [Persister](persisters/api.md) instance.

```php
$persister = new Filesystem('example.transformer');

$transformer = $persister->load();

$dataset->apply($transformer);
```

## Transformer Pipelines
The [Pipeline](pipeline.md) meta-estimator helps you automate a series of transformations applied to the input dataset to an estimator. With a Pipeline, any dataset object passed to will automatically be fitted and/or transformed before it arrives in the estimator's context. In addition, transformer fittings can be saved alongside the model data when the Pipeline is persisted.

```php
use Rubix\ML\Pipeline;
use Rubix\ML\Transformers\RandomHotDeckImputer;
use Rubix\ML\Transformers\OneHotEncoder;
use Rubix\ML\Transformers\ZScaleStandardizer;
use Rubix\ML\Clusterers\KMeans;

$estimator = new Pipeline([
    new RandomHotDeckImputer(5),
    new OneHotEncoder(),
    new ZScaleStandardizer(),
], new KMeans(10, 256));
```

Calling `train()` or `partial()` will result in the transformers being fitted or updated before being passed to the Softmax Classifier.

```php
$estimator->train($dataset); // Transformers fitted and applied

$estimator->partial($dataset); // Transformers updated and applied
```

Any time a dataset is passed to the Pipeline it will automatically be transformed before being handed to the underlying estimator.

```php
$predictions = $estimator->predict($dataset); // Dataset transformed automatically
```

You can save the transformer fittings alongside the model data by persisting the entire Pipeline object or by wrapping it in a [Persistent Model](persistent-model.md) meta-estimator.

```php
$persister = new Filesystem('example.model');

$persister->save($estimator);
```

## Advanced Preprocessing
In some cases, certain features of a dataset may require a different set of preprocessing steps than the others. In such a case, we are able to extract only certain features, preprocess them, and then join them to another set of features. In the example below, we'll extract just the text reviews and their sentiment labels into a dataset object and put the sample's category, number of clicks, and ratings into another one using two [Column Pickers](extractors/column-picker.md). Then, we can apply a separate set of transformations to each set of features and use the `join()` method to combine them into a single dataset. We can even apply another set of transformations to the dataset after that.

```php
use Rubix\ML\Dataset\Labeled;
use Rubix\ML\Extractors\ColumnPicker;
use Rubix\ML\Extractors\NDJSON;
use Rubix\ML\Dataset\Unlabeled;
use Rubix\ML\Transformers\TextNormalizer;
use Rubix\ML\Transformers\WordCountVectorizer;
use Rubix\ML\Transformers\TfIdfTransformer;
use Rubix\ML\Transformers\OneHotEncoder;
use Rubix\ML\Transformers\ZScaleStandardizer;

$extractor1 = new ColumnPicker(new NDJSON('dataset.ndjson'), [
    'review', 'sentiment',
]);

$extractor2 = new ColumnPicker(new NDJSON('dataset.ndjson'), [
    'category', 'clicks', 'rating',
]);

$dataset1 = Labeled::fromIterator($extractor1)
    ->apply(new TextNormalizer())
    ->apply(new WordCountVectorizer(5000))
    ->apply(new IfIdfTransformer());

$dataset2 = Unlabeled::fromIterator($extractor2)
    ->apply(new OneHotEncoder());

$dataset = $dataset1->join($dataset2)
    ->apply(new ZScaleStandardizer());
```

## Filtering Records
In some cases, you may want to remove entire rows from the dataset. For example, you may want to remove records that contain features with abnormally low/high values as these samples can be interpreted as noise. The `filterByColumn()` method on the dataset object uses a callback function to determine whether or not to return a row in the new dataset by the value of the feature at a given column offset.

```php
$tallPeople = $dataset->filterByColumn(3, function ($value) {
	return $value > 178.5;
});
```

## De-duplication
When it is undesirable for a dataset to contain duplicate records, you can remove all duplicates by calling the `deduplicate()` method on the dataset object.

```php
$dataset->deduplicate();
```

!!! note
    The O(N^2) time complexity of de-duplication may be prohibitive for large datasets.

## Saving a Dataset
If you ever want to preprocess a dataset and then save it for later you can do so by calling one of the conversion methods (`toCSV()`, `toNDJSON()`) on the [Dataset](datasets/api.md#encode-the-dataset) object. Then, call the `write()` method on the returned encoding object to save the data to a file at a given path like in the example below.

```php
use Rubix\ML\Transformers\MissingDataImputer;

$dataset->apply(new MissingDataImputer())->toCSV()->write('dataset.csv');
```
