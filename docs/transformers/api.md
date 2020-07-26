# Transformer
Transformers take Dataset objects and modify the features contained within. They are often used as part of a transformer [Pipeline](../pipeline.md) or they can be used on their own.

### Transform a Dataset
To transform a dataset, pass a transformer object to the `apply()` method on a [Dataset](../datasets/api.md) object like in the example below.

```php
use Rubix\ML\Transformers\MinMaxNormalizer;

$dataset->apply(new MinMaxNormalizer());
```

The transformer can directly transform the samples in place via the `transform()` method given a samples array:
```php
public transform(array &$samples) : void
```

```php
$transformer->transform($samples);
```

## Stateful
Stateful transformers are those that require *fitting* before they can transform. The `fit()` method takes a dataset as input and pre-computes any necessary information in order to carry out future transformations. You can think of *fitting* a transformer like *training* a learner.

### Fit a Dataset
To fit the transformer to a training set:
```php
public fit(Dataset $dataset) : void
```

Check if the transformer has been fitted:
```php
public fitted() : bool
```

```php
use Rubix\ML\Transformers\OneHotEncoder;

$transformer = new OneHotEncoder();

$transformer->fit($dataset);

var_dump($transformer->fitted());
```

```sh
bool(true)
```

To apply a Stateful transformer to a dataset object, pass the transformer instance to the `apply()` method like you normally would. The transformer will automatically be fitted with the dataset before transforming the samples.

```php
use Rubix\ML\Transformers\OneHotEncoder;

$dataset->apply(new OneHotEncoder());
```

## Elastic
Some transformers are able to adapt to new training data. The `update()` method provided by the Elastic interface can be used to modify the fitting of the transformer with new data even after being previously fitted. *Updating* is the transformer equivalent to partially training an Online learner.

### Update a Fitting
```php
public update(Dataset $dataset) : void
```

```php
use Rubix\ML\Transformers\ZScaleStandardizer;

$transformer = new ZScaleStandardizer();

$folds = $dataset->fold(3);

$transformer->fit($folds[0]);

$transformer->update($folds[1]);

$transformer->update($folds[2]);
```