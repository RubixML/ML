# Representing Your Data
A dataset is made up of a matrix of *samples* comprised of *features* which are usually scalar values. Each sample is a sequentially-numbered array with exactly the same number of elements as the rest. The columns of the matrix contain the values for the particular feature represented by that column. The *dimensionality* of a sample is equal to the number of features it has. For example, the samples below are said to be *3-dimensional* because they contain 3 feature columns. You'll notice that samples can be made up of a heterogeneous mix of data types which we'll describe in detail in the next sections.

```php
$samples = [
    [0.1, 21.5, 'furry'],
    [2.0, -5, 'rough'],
    [0.001, -10, 'rough'],
];
```

## High-level Data Types
The library comes with a higher-order type system which distinguishes types that are continuous (numerical), categorical (discrete), or some other data type. Continuous features represent some *quantitative* property of the sample such as *age* or *velocity*, whereas categorical features form a *qualitative* property such as *rough* or *furry*.

| Rubix ML Data Type | PHP Internal Type |
|---|---|
| Continuous | Integer or Float |
| Categorical | String |
| Image | GD Resource |

## Quantities
A quantity is a property that describes either the magnitude or multitude of something. For example, *temperature*, *income*, and *age* are all quantitative features. In Rubix ML, quantities are represented as one of the continuous data types such as integers or floating point numbers and their *distances* are assumed to be equally-spaced. Quantities can further be broken down into ratios, intervals, or counts depending on the feature they are describing.

## Categories
Categories are discrete values that describe some qualitative property of a sample such as *species*, *gender*, or *nationality*. They are represented as strings and have no numerical relationship between the values. Unlike ratios and intervals, which can take on an infinite number of values, categorical variables can only take on 1 of a finite set of values.

## Booleans
A boolean (or *binary*) variable is a special case of a categorical variable in which the number of possible categories is strictly two. For example, to denote if a subject is tall or not you can use the `tall` and `not tall` categories respectively.

## Ordinals
Even though PHP treats numeric strings like `'1'` and `'2'` as if they were numeric, they are still considered categorical variables in Rubix ML. This conveniently allows you to represent ordinal variables as *ordered categories*. For example, instead of the integers `1`, `2`, `3`, `...`, which imply a precise interval, you could use the strings `'1'`, `'2'`, `'3'`, `...` to signal ordinal values in which the distances between values could be arbitrary.

## Date/Time
There are a number of ways datetime features can be represented in a dataset. One way is to discretize the value into days, months, and/or years using categories like `1`, `2`, `3`, `...`, `june`, `july`, `august`, and `2019`, `2020` etc. Datetime features can also be represented as continuous by converting them to integer-valued timestamps.

## Text
Text data are a product of language communication and can be viewed as an encoding of many individual features. Initially, text blobs are imported as categorical, however, they have little meaning as a category because the features are still encoded. Thus, import text blobs as simple strings and use a [preprocessing](preprocessing.md) step to extract features such as word counts, weighted term frequencies, or word embeddings.

## Images
Images are represented as the [GD](https://www.php.net/manual/en/book.image.php) resource type. A resource is a special variable that holds a reference to some external data such as an image file. For this reason, resources must eventually be converted into a scalar type before use with a learner. In the case of images, they will most often be converted to raw color channel data by reading the RGB intensities of each pixel.

## What about NULL?
Null values are used to indicate the absence of a value, however since it does not give any information as to the *type* of the variable that is missing, it cannot be used in a dataset. Instead, represent missing values as either `NaN` for continuous features or use a separate category (such as `?`) to denote missing categorical values.