# Obtaining Data
Machine learning projects typically begin with a question. For example, you might want to answer the question "who of my friends are most likely to stay married to their partner?" One way to go about answering this question with machine learning would be to go out and ask a bunch of happily married and divorced couples the same set of questions about their partner and then use that data to build a model to predict successful or doomed relationships based on the answers your friends give you. In ML terms, the answers you collect are called *features* and they constitute measurements of some phenomena being observed. The number of features in a sample is called the *dimensionality* of the sample. For example, a sample with 20 features is said to be *20 dimensional*.

An alternative to collecting data yourself is to download one of the many open datasets that are free to use from a public repository. The advantages of using a public dataset is that, usually, the data has already been cleaned and prepared for you. We recommend the University of California Irvine [Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets.php) as a great place to get started with using open source datasets.

## Extracting Data
Before data can become useful, we need to load it into the computer in the proper format. This involves extracting the data from source before anything else. There are many PHP libraries that help make extracting data from various sources easy and intuitive, and we recommend checking them out as a great place to start.

- [PHP League CSV](https://csv.thephpleague.com/) - Generator-based CSV extractor
- [Doctrine DBAL](https://www.doctrine-project.org/projects/dbal.html) - SQL database abstraction layer
- [Google BigQuery](https://cloud.google.com/bigquery/docs/reference/libraries) - Cloud-based data warehouse via SQL

## The Dataset Object
In Rubix, data is passed around in specialized data containers called Dataset objects. [Datasets](#dataset-objects) internally handle selecting, splitting, folding, transforming, and randomizing the samples and labels contained within. In general, there are two types of datasets, *Labeled* and *Unlabeled*. Labeled datasets are used for supervised learning and for providing the ground-truth during cross validation. Unlabeled datasets are used for unsupervised learning and for making predictions (*inference*) on unknown samples.

As a simplistic example, suppose that you went out and asked 100 couples (50 married and 50 divorced) to rate their partner's communication skills (between 1 and 5), attractiveness (between 1 and 5), and time spent together per week (hours per week). You could construct a [Labeled](#labeled) Dataset object from this data by passing the samples and labels into the constructor.

```php
use Rubix\ML\Datasets\Labeled;

$samples = [
    [3, 4, 50.5], [1, 5, 24.7], [4, 4, 62.0], [3, 2, 31.1]
];

$labels = ['married', 'divorced', 'married', 'divorced'];

$dataset = new Labeled($samples, $labels);
```