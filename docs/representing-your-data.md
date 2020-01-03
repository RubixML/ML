# Representing Your Data
Data are a first class citizen in Rubix ML. The library makes it easy to work with datasets through its [Dataset](./datasets/api.md) object, which is a specialized container for data that every learner can recognize. Dataset objects allow you to validate, sort, randomize, split, fold, and describe your data among many other things. The most basic dataset is made up of a data table (or *matrix*) of samples comprised of *features* which are usually scalar variables. Each sample (row) is a sequential array with exactly the same number of elements as the rest. The column of a sample represents the value of a particular feature. The *dimensionality* of a sample is equal to the number of features it has. For example, the samples below are said to be *3-dimensional* because they contain 3 feature columns. You'll notice that samples can be made up of a heterogeneous mix of data types which we'll describe in detail in the next sections.

```php
$samples = [
    [0.1, 21.5, 'furry'],
    [2.0, -5, 'rough'],
    [0.001, -10, 'rough'],
];
```

## High-level Data Types
In addition to PHP's internal type system, Rubix ML adds a layer on top which distinguishes types that are continuous (numerical), categorical (discrete), or some other type such as resource. Continuous features represent some *quantitative* property of the sample such as age or velocity, whereas, categorical features form a *qualitative* property such as rough or furry. We make this distinction because different learners are compatible with different data types. For example, the [Naive Bayes](./classifiers/naive-bayes.md) classifier is compatible with only categorical features but [Gaussian Naive Bayes](./classifiers/gaussian-naive-bayes.md) is compatible with continuous - and [Random Forest](./classifiers/random-forest.md) is compatible with both.

| Rubix ML Type | Internal PHP Type |
|---|---|
| Continuous | integer or float |
| Categorical | string |
| Resource | resource |
| Other | object, bool, null, etc. |

## Quantities
A quantity is a property that describes either the magnitude or multitude of something. For example, temperature, income, and age are all quantitative features. In Rubix ML, quantities are represented as one of the continuous data types such as integers or floating point numbers and their *distances* are assumed to be equally-spaced. For example, the distance between 10 years old and 11 is exactly 1 year. Quantities can further be broken down into ratios, intervals, or counts depending on the feature they are describing.

## Categories
Categories are discrete values that describe some qualitative property of a sample such as `species`, `gender`, or `nationality`. They are represented as strings and have no numerical relationship between the values. Unlike ratios and intervals, which can take on an infinite number of values, categorical variables can only take on 1 of *k* discrete values.

## Booleans
A boolean (or *binary*) variable is a special case of a categorical variable when the number of possible categories is strictly two. For example, to denote if a subject is tall or not you can use either the `tall` or `not tall` categories respectively.

## Ordinals
Even though PHP treats numeric strings such as `'1'` and `'2'` as if they were numeric, they are still considered categorical variables according to Rubix ML's high-level type system. This conveniently allows you to represent ordinal variables as *ordered categories*. For example, instead of the integers `1`, `2`, `3`, `...`, which imply a precise interval, you could use the strings `'1'`, `'2'`, `'3'`, `...` to signal ordinal values in which the distances between values could be arbitrary.

## Date/Time
There are a few ways that datetime features can be represented in a dataset. One way is to represent the days, months, and/or years categorically using discrete values such as `june`, `july`, `august`, or `2019`, `2020`. A datetime can also be represented as a continuous feature by converting it to an integer timestamp.

## Images
Images are represented as the resource type both in PHP and according to Rubix ML's high-level type system. A resource is a special variable that holds a reference to some external data such as an image file. For this reason, resources must eventually be converted into a scalar type for compatibility with a learner. In the case of images, they will most often be converted to raw color channel data by reading the RGB values of each pixel.

## Text
Text data are a product of a process called language communication and can be viewed as an encoding of many individual features. Initially, each blob is imported as a single categorical feature, however, they have little meaning as a category because the features are still encoded in the language. Thus, import text blobs as simple strings and use a [preprocessing](preprocessing.md) step to extract features such as word counts, weighted term frequencies, or word embeddings.

## What about NULL?
Null values are often used to indicate the absence of a value, however since it does not give any information as to the type of the variable that is missing, it cannot be used in a dataset. Instead, represent missing values as either `NaN` for continuous variables or use a separate category (such as `?`) to denote missing categorical values.