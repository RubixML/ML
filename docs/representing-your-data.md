# Representing Your Data
The library makes it easy to work with your data via the [Dataset](datasets/api.md) object, which is a specialized data container that every learner can recognize. A dataset is made up of a matrix of *samples* comprised of *features* which are usually scalar values. Each sample is a sequentially-ordered array with exactly the same number of elements as every other sample. The columns of the matrix contain the values for a particular feature represented by that column. The *dimensionality* of a sample is equal to the number of features it has. For example, the samples below are said to be *3-dimensional* because they contain 3 feature columns. You'll notice that samples can be made up of a heterogeneous mix of data types which we'll describe in detail in the next sections.

```php
$samples = [
    [0.1, 21.5, 'furry'],
    [2.0, -5, 'rough'],
    [0.001, -10, 'rough'],
];
```

## High-level Data Types
The library comes with a built-in higher-order type system which distinguishes types that are continuous (numerical), categorical (discrete), or some other data type. Continuous features represent some *quantitative* property of the sample such as temperature or credit limit, whereas categorical features form a *qualitative* property of a sample such as color or texture.

| Rubix Data Type | PHP Internal Type |
|---|---|
| Categorical | String |
| Continuous | Integer or Floating Point Number |
| Image | GD Resource |

## Quantities
A quantity is a property that describes the magnitude or multitude of something. Temperature, credit limit, and age are all quantitative features. In Rubix ML, quantities are represented as one of the continuous data types such as integers or floating point numbers. Quantities can further be broken down into ratios, intervals, or counts and there is a measurable numerical relationship between their values.

## Categories
Categories are discrete values that describe a qualitative property of a sample such as color, texture, or personality type. They are represented as strings and have no numerical relationship between the values. Unlike some quantities that can take on an infinite number of values, categorical variables can only take on 1 of a finite set of values.

## Booleans
A boolean (or *binary*) variable is a special case of a categorical variable in which the number of possible categories is strictly two. For example, to denote if a subject is tall or not you can use the `tall` and `not tall` categories respectively.

## Ordinals
Even though PHP treats numeric strings like `'1'` and `'2'` as if they were numeric, they are still considered categorical variables in Rubix ML. This conveniently allows you to represent ordinal variables as *ordered categories*. For example, instead of the integers `1`, `2`, `3`, `...`, which imply a precise interval, you could use the strings `'1'`, `'2'`, `'3'`, `...` to signal ordinal values in which the distances between values could be arbitrary.

## Date/Time
There are a number of ways that date/time features can be represented in a dataset. One way is to discretize the value into days, months, and years using categories like `1`, `2`, `3`, `...`, `june`, `july`, `august`, and `2019`, `2020` etc. Date/times can also be represented as continuous features by converting them to a single numerical timestamp.

## Text
Text data are a product of language communication and can be viewed as an encoding of many individual features. Initially, text blobs are imported as categorical features, however, they have little meaning as a category because the features are still encoded. Thus, import text blobs and use a [preprocessing](preprocessing.md) step to extract features such as word counts, weighted term frequencies, or word embeddings.

## Images
Images are represented as either the [GD](https://www.php.net/manual/en/book.image.php) resource type or (as of PHP 8) a `GdImage` object. An image type is a special type that holds a reference to the data stored within the image file. For this reason, images must eventually be converted to a scalar type, such as the RGB color intensity values of each pixel, before they can be the input to a learning algorithm.

## What about NULL?
Null values are often used to indicate the absence of a value, however since they do not give any information as to the type of variable that is missing, they cannot be used in a dataset. Instead, represent missing values as either the standard PHP math constant `NAN` for continuous features or use a special category (such as `?`) to denote missing categorical values.
