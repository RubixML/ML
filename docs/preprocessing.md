# Preprocessing
Sometimes, one or more preprocessing steps may need to be taken before handing a dataset off to a Learner. In some cases, data may not be in the correct format and in others you may want to process the data to aid in training.

## Transformers
[Transformers](transformers/api.md) are objects that perform various preprocessing steps to the samples in a dataset. They take a dataset object as input and transform it in place. [Stateful](transformers/api.md#stateful) transformers are a type of transformer that must be *fitted* to a dataset. Fitting a dataset to a transformer is much like training a learner but in the context of preprocessing rather than inference. After fitting a stateful transformer, it will expect the features to be present in the same order when transforming subsequent datasets. A few transformers are *supervised* meaning they must be fitted with a [Labeled](datasets/labeled.md) dataset. [Elastic](transformers/api.md#elastic) transformers can have their fittings updated with new data after an initial fitting.

### Transform a Dataset
An example of a transformation is one that converts the categorical features of a dataset to continuous ones using a [*one hot*](https://en.wikipedia.org/wiki/One-hot) encoding. To accomplish this with the library, pass a [One Hot Encoder](transformers/one-hot-encoder.md) instance as an argument to the [Dataset](datasets/api.md) object's `apply()` method. Note that the `apply()` method also handles fitting a Stateful transformer automatically.

```php
use Rubix\ML\Transformers\OneHotEncoder;

$dataset->apply(new OneHotEncoder());
```

Transformations can be chained by calling the `apply()` method fluently.

```php
use Rubix\ML\Transformers\HotDeckImputer;
use Rubix\ML\Transformers\OneHotEncoder;
use Rubix\ML\Transformers\MinMaxNormalizer;

$dataset->apply(new HotDeckImputer(5))
    ->apply(new OneHotEncoder())
    ->apply(new MinMaxNormalizer());
```

### Transforming the Labels
Transformers do not alter the labels in a dataset. For that we can pass a callback function to the `transformLabels()` method on a [Labeled](datasets/labeled.md#transform-labels) dataset instance. The callback accepts a single argument that is the value of the label to be transformed. In this example, we'll convert the categorical labels of a dataset to integer ordinals.

```php
$dataset->transformLabels('intval');
```

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

## Types of Preprocessing
Here we dive into the different types of data preprocessing that Transformers are capable of.

### Standardization and Normalization
Oftentimes, the continuous features of a dataset will be on different scales because they were measured by different methods. For example, age (0 - 100) and income (0 - 9,999,999) are on two widely different scales. Standardization is the processes of transforming a dataset such that the features are all on one common scale. Normalization is the special case where the transformed features have a range between 0 and 1. Depending on the transformer, it may operate on the columns or the rows of the dataset.

| Transformer | Operates | Output Range | [Stateful](transformers/api.md#stateful) | [Elastic](transformers/api.md#elastic) |
|---|---|---|---|---|
| [L1 Normalizer](transformers/l1-normalizer.md) | Row-wise | [0, 1] | | |
| [L2 Normalizer](transformers/l2-normalizer.md) | Row-wise | [0, 1] | | |
| [Max Absolute Scaler](transformers/max-absolute-scaler.md) | Column-wise | [-1, 1] | ● | ● |
| [Min Max Normalizer](transformers/min-max-normalizer.md) | Column-wise | [min, max] | ● | ● |
| [Robust Standardizer](transformers/robust-standardizer.md) | Column-wise | [-∞, ∞] | ● | |
| [Z Scale Standardizer](transformers/z-scale-standardizer.md) | Column-wise | [-∞, ∞] | ● | ● |

### Feature Conversion
Feature converters are transformers that convert feature columns of one data type to another by changing their representation.

| Transformer | From | To | [Stateful](transformers/api.md#stateful) | [Elastic](transformers/api.md#elastic) |
|---|---|---|---|---|
| [Interval Discretizer](transformers/interval-discretizer.md) | Continuous | Categorical | ● | |
| [One Hot Encoder](transformers/one-hot-encoder.md) | Categorical | Continuous | ● | |
| [Numeric String Converter](transformers/numeric-string-converter.md) | Categorical | Continuous | | |
| [Boolean Converter](transformers/boolean-converter.md) | Other | Categorical or Continuous | | |

### Dimensionality Reduction
Dimensionality reduction is a preprocessing technique for projecting a dataset onto a lower dimensional vector space. It allows a learner to train and infer quicker by producing a training set with fewer but more informative features. Dimensionality reducers can also be used to visualize datasets by outputting low (1 - 3) dimensionality embeddings for use in plotting software.

| Transformer | Supervised | [Stateful](transformers/api.md#stateful) | [Elastic](transformers/api.md#elastic) |
|---|---|---|---|
| [Gaussian Random Projector](transformers/gaussian-random-projector.md) | | ● | |
| [Linear Discriminant Analysis](transformers/linear-discriminant-analysis.md) | ● | ● | |
| [Principal Component Analysis](transformers/principal-component-analysis.md) | | ● | |
| [Sparse Random Projector](transformers/sparse-random-projector.md) | | ● | |
| [Truncated SVD](transformers/truncated-svd.md) | | ● | |
| [t-SNE](transformers/t-sne.md) | | | |

### Feature Expansion
Feature expansion aims to add flexibility to a model by deriving additional features from a dataset. It can be thought of as the opposite of dimensionality reduction.

| Transformer | Supervised | [Stateful](transformers/api.md#stateful) | [Elastic](transformers/api.md#elastic) |
|---|---|---|---|
| [Polynomial Expander](transformers/polynomial-expander.md) | | | |

### Imputation
Imputation is a technique for handling missing values in a dataset by replacing them with a pretty good guess.

| Transformer | Compatibility | Supervised | [Stateful](transformers/api.md#stateful) | [Elastic](transformers/api.md#elastic) |
|---|---|---|---|---|
| [KNN Imputer](transformers/knn-imputer.md) | Depends on distance kernel | | ● | |
| [Missing Data Imputer](transformers/missing-data-imputer.md) | Categorical, Continuous | | ● | |
| [Hot Deck Imputer](transformers/hot-deck-imputer.md) | Depends on distance kernel | | ● | |

### Natural Language
The library provides a number of transformers for Natural Language Processing (NLP) and Information Retrieval (IR) tasks such as text cleaning, feature extraction, and term weighting.

| Transformer | Supervised | [Stateful](transformers/api.md#stateful) | [Elastic](transformers/api.md#elastic) |
|---|---|---|---|
| [BM25 Transformer](transformers/bm25-transformer.md) | | ● | ● |
| [Regex Filter](transformers/regex-filter.md) | | | |
| [Text Normalizer](transformers/text-normalizer.md) | | | |
| [Multibyte Text Normalizer](transformers/multibyte-text-normalizer.md) | | | |
| [Stop Word Filter](transformers/stop-word-filter.md) | | | |
| [TF-IDF Transformer](transformers/tf-idf-transformer.md) | | ● | ● |
| [Token Hashing Vectorizer](transformers/token-hashing-vectorizer.md) | | | |
| [Word Count Vectorizer](transformers/word-count-vectorizer.md) | | ● | |

### Images
These transformers operate on the high-level image data type.

| Transformer | Supervised | [Stateful](transformers/api.md#stateful) | [Elastic](transformers/api.md#elastic) |
|---|---|---|---|
| [Image Resizer](transformers/image-resizer.md) | | | |
| [Image Rotator](transformers/image-rotator.md) | | | |
| [Image Vectorizer](transformers/image-vectorizer.md) | | ● | |

## Custom Transformations
In additional to providing specialized Transformers for common preprocessing tasks, the library includes a [Lambda Function](transformers/lambda-function.md) transformer that allows you to apply custom data transformations using a callback. The callback function accepts a sample passed by reference so that the transformation occurs in-place. In the following example, let's write a callback to *binarize* the continuous features just at column offset 3 of the dataset.

```php
use Rubix\ML\Transformers\LambdaFunction;

$binarize = function (&$sample) {
    $sample[3] = $sample[3] > 182 ? 'tall' : 'not tall';
}

$dataset->apply(new LambdaFunction($binarize));
```

Another technique we can employ using the Lambda Function transformer is to perform a categorical *feature cross* between two feature columns of a dataset. A cross feature is a higher-order feature that represents the presence of two or more features simultaneously. For example, we may want to represent the combination of someone's gender and education level as it's own feature. We'll choose to represent the new feature as a [CRC32](https://en.wikipedia.org/wiki/Cyclic_redundancy_check) hash to save on memory and storage but you could just concatenate both categories to represent the new feature as well.

```php
use Rubix\ML\Transformers\LambdaFunction;
use function hash;

$crossFeatures = function (&$sample) {
    $sample[] = hash('crc32b', "{$sample[6]} and {$sample[7]}");
};

$dataset->apply(new LambdaFunction($crossFeatures));
```

## Advanced Preprocessing
In some cases, certain features of a dataset may require a different set of preprocessing steps than the others. In such a case, we can extract a certain set of features, preprocess them, and then join them with the rest of the dataset later. In the example below, we'll extract just the text reviews and their sentiment labels into a dataset object and put the sample's category, number of clicks, and ratings into another one using two [Column Pickers](extractors/column-picker.md). Then, we can apply a separate set of transformations to each set of features and use the `join()` method to combine them into a single dataset. We can even apply another set of transformations to the joined dataset after that.

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

$extractor1 = new ColumnPicker(new NDJSON('example.ndjson'), [
    'review', 'sentiment',
]);

$extractor2 = new ColumnPicker(new NDJSON('example.ndjson'), [
    'category', 'clicks', 'rating',
]);

$dataset1 = Labeled::fromIterator($extractor1)
    ->apply(new TextNormalizer())
    ->apply(new WordCountVectorizer(5000))
    ->apply(new TfIdfTransformer());

$dataset2 = Unlabeled::fromIterator($extractor2)
    ->apply(new OneHotEncoder());

$dataset = $dataset1->join($dataset2)
    ->apply(new ZScaleStandardizer());
```

## Transformer Pipelines
The [Pipeline](pipeline.md) meta-estimator helps you automate a series of transformations applied to the input dataset to an estimator. With a Pipeline, any dataset object passed to will automatically be fitted and/or transformed before it arrives in the estimator's context. In addition, transformer fittings can be saved alongside the model data when the Pipeline is persisted.

```php
use Rubix\ML\Pipeline;
use Rubix\ML\Transformers\HotDeckImputer;
use Rubix\ML\Transformers\OneHotEncoder;
use Rubix\ML\Transformers\ZScaleStandardizer;
use Rubix\ML\Clusterers\KMeans;

$estimator = new Pipeline([
    new HotDeckImputer(5),
    new OneHotEncoder(),
    new ZScaleStandardizer(),
], new KMeans(10));
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

## Filtering Records
In some cases, you may want to remove entire rows from the dataset. For example, you may want to remove records that contain features with abnormally low/high values as these samples can be interpreted as noise. The `filter()` method on the dataset object uses a callback function to determine if a row should be included in the return dataset. In this example, we'll filter all the samples whose value for feature at offset 3 is greater than some amount.

```php
$tallPeople = function ($record) {
	return $record[3] > 178.5;
};

$dataset = $dataset->filter($tallPeople);
```

Let's say we wanted to train a classifier with our [Labeled](datasets/labeled.md) dataset but only on a subset of the possible class outcomes. We could filter the samples that correspond to undesirable outcomes by targetting the label with our callback.

```php
use function in_array;

$dogsAndCats = function ($record) {
    return in_array(end($record), ['dog', 'cat']);
}

$training = $dataset->filter($dogsAndCats);
```

!!! note
    For [Labeled](datasets/labeled.md) datasets the label column is always the last column of the record.

In the next example, we'll filter all the records that have missing feature values. We can detect missing continuous variables by calling the custom library function `iterator_contains_nan()` on each record. Additionally, we can filter records with missing categorical values by looking for a special placeholder category, in this case we'll use the value `'?'`, to denote missing categorical variables.

```php
use function Rubix\ML\iterator_contains_nan;
use function in_array;

$noMissingValues = function ($record) {
    return !iterator_contains_nan($record) and !in_array('?', $record);
};

$complete = $dataset->filter($noMissingValues);
```

!!! note
    The standard PHP library function `in_array()` does not handle `NAN` comparisons.

## De-duplication
When it is undesirable for a dataset to contain duplicate records, you can remove all duplicates by calling the `deduplicate()` method on the dataset object.

```php
$dataset->deduplicate();
```

!!! note
    The O(N^2) time complexity of de-duplication may be prohibitive for large datasets.
