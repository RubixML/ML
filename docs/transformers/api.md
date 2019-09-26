# Transformer
Transformers take Dataset objects and apply transformations to the features contained within the samples. They are often used as part of a [Pipeline](../pipeline.md) or they can be used standalone.

### Transform Samples
The transformer directly transforms the samples in place via the `transform()` method:
```php
public transform(array &$samples) : void
```

**Example**

```php
use Rubix\ML\Transformers\NumericStringConverter;

// Import samples

$transformer = new NumericStringConverter();

$transformer->transform($samples);
```

To transform a dataset without having to pass the raw samples, pass a transformer object to the `apply()` method on a [Dataset](../datasets/api.md) object.

**Example**

```php
$dataset->apply(new NumericStringConverter());
```

# Stateful
For stateful transformers, the `fit()` method will allow the transformer to compute any necessary information from the training set in order to carry out its future transformations. You can think of *fitting* a transformer like *training* a learner.

### Fit a Dataset
To fit the transformer to a training set:
```php
public fit(Dataset $dataset) : void
```

Check if the transformer has been fitted:
```php
public fitted() : bool
```

**Example**

```php
use Rubix\ML\Transformers\OneHotEncoder;

$transformer = new OneHotEncoder();

$transformer->fit($dataset);

var_dump($transformer->fitted());
```

```sh
bool(true)
```

To fit and apply a Stateful transformer to a dataset object at the same time, simply pass the transformer instance to the `apply()` method.

```php
$dataset->apply(new OneHotEncoder());
```

# Elastic
Some transformers are able to adapt to new training data. The `update()` method on transformers that implement the Elastic interface can be used to modify the fitting of the transformer with new data even after it has previously been fitted. *Updating* is the transformer equivalent to *partially training* an online learner.

### Update a Fitting
```php
public update(Dataset $dataset) : void
```

**Example**

```php
use Rubix\ML\Transformers\ZScaleStandardizer;

$transformer = new ZScaleStandardizer();

$folds = $dataset->fold(3);

$transformer->fit($folds[0]);

$transformer->update($folds[1]);

$transformer->update($folds[2]);
```