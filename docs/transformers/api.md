### Transformers
Transformers take [Dataset](#dataset-objects) objects and apply blanket transformations to the features contained within them. They are often used as part of a [Pipeline](#pipeline) or they can be used by themselves. Examples of transformations are scaling and centering, normalization, dimensionality reduction, missing data imputation, and feature selection.

The transformer directly transforms the samples in place via the `transform()` method:
```php
public transform(array &$samples) : void
```

> **Note**: To transform a dataset without having to pass the raw sample, instead you can pass a transformer object to the `apply()` method on a Dataset object.

### Stateful
For stateful transformers, the `fit()` method will allow the transformer to compute any necessary information from the training set in order to carry out its future transformations. You can think of *fitting* a transformer like *training* a learner.

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
```

### Elastic
Some transformers are able to adapt to new training data. The `update()` method on transformers that implement the Elastic interface can be used to modify the fitting of the transformer with new data even after it has previously been fitted. *Updating* is the transformer equivalent to *partially training* an online learner.

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