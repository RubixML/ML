# Exploring Data
Exploratory Data Analysis (EDA) is an approach to modeling data that produces insights from their summarization and visualization. EDA is often used in the engineering of features as well as model selection and can save time and lead to more accurate models when included in your machine learning lifecycle. In general, there are two types of Exploratory Data Analysis - quantitative and graphical. Quantitative analysis summarizes the data using statistical or probabilistic methods that are then interpreted for meaning. Graphical analysis uses techniques such as scatterplots and histograms to glean information from the structure and shape of the data and often incorporates Manifold Learning to reduce the dimensionality of the samples. 

## Describe a Dataset
The Dataset API has a handy method named `describe()` that computes statistics for each continuous feature of the dataset such as the column median, standard deviation, and skewness. In addition, it provides the probabilities of each category for categorical feature columns. In the example below, we'll echo the Report object returned by the `describe()` method to get a better understanding for how the values of our features are distributed.

```php
$report = $dataset->describe();

echo $report;
```

```json
[
    {
        "offset": 0,
        "type": "categorical",
        "num categories": 2,
        "probabilities": {
            "friendly": 0.6666666666666666,
            "loner": 0.3333333333333333
        }
    },
    {
        "offset": 1,
        "type": "continuous",
        "mean": 0.3333333333333333,
        "variance": 9.792222222222222,
        "stddev": 3.129252661934191,
        "skewness": -0.4481030843690633,
        "kurtosis": -1.1330702741786107,
        "min": -5,
        "25%": -1.375,
        "median": 0.8,
        "75%": 2.825,
        "max": 4
    }
]
```

We can also save the report by passing a [Persister](persisters/api.md) to the `saveTo()` method on the Encoding object returned by calling `toJSON()` on the Report object.

```php
use Rubix\ML\Persisters\Filesystem;

$report->toJSON()->saveTo(new Filesystem('example.json'));
```

### Describe by Label


## Plotting

On the map ...

## Manifold Learning
Manifold Learning is a type of dimensionality reduction that aims to produce a faithful low-dimensional (1 - 3) embedding of a dataset for visualization. In the example below, we'll use [t-SNE](transformers/t-sne.md) to embed the dataset into 2 dimensions and then save the data to a CSV file so we can import it into our plotting software.

```php
use Rubix\ML\Transformers\TSNE;
use Rubix\ML\Extractors\CSV;

$dataset->apply(new TSNE(2))->writeTo(new CSV('example.csv'));
```
