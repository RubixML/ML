# Representing Your Data
The library makes it easy to work with your data via the [Dataset](datasets/api.md) object, which is a specialized data container that every learner can recognize. A dataset is made up of a matrix of *samples* comprised of *features* which are usually scalar values. Each sample is a sequentially-ordered array with exactly the same number of elements as every other sample. The columns of the matrix contain the values for a particular feature represented by that column. The *dimensionality* of a sample is equal to the number of features it has. For example, the samples below are said to be *3-dimensional* because they contain 3 feature columns. You'll notice that samples can be made up of a heterogeneous mix of data types.

```php
$samples = [
    [0.1, 21.5, 'furry'],
    [2.0, -5, 'rough'],
    [0.001, -10, 'rough'],
];
```

## High-level Data Types
The library comes with a higher-order type system that distinguishes types that are continuous, categorical (discrete), or some other data type. The distinction between types is important for determining the operations that can be performed on a particular feature.

| Library Type | PHP Type |
|---|---|
| Continuous | Integer or floating point number |
| Categorical | String |
| Image | GD Image object or resource |

## Continuous Features
Continuous features represent some *quantitative* property of the sample and are represented as natural, integer, or real (floating point) numbers. They can be broken down into intervals, ratios, and counts each with their own properties and constraints. One property they all share, however, is that the distances between adjacent values are equal and consistent.

### Intervals
Intervals are the most general form of continuous measurement and can take on any value within the set of real numbers. Some examples of interval data include temperature in Celsius or Fahrenheit, income, and scores on a personality test.

### Ratios
Ratios are lower bounded at a fixed zero point. Due to this extra constraint, ratio variables are able to say something about the relative differences between samples by comparing numbers on the scale to absolute zero. Examples of ratio data include height, distance, and temperature in Kelvin.
### Counts
Count variables are limited to the set of natural (or *counting*) numbers and therefore are always non-negative.

## Categorical Features
Categories are discrete values that describe a qualitative property of a sample such as texture, genre, or political party. They are represented as strings and, unlike continuous features, have no numerical relationship between the values.

### Categories
Categorical or *nominal* variables specify which category a sample belongs to among a finite set of choices. For example, a texture feature might include categories such as `rough`, `furry`, or `smooth`.

### Ordinals
Numeric strings such as `'1'` and `'2'` are considered categorical variables in our high-level type system. This conveniently allows you to represent ordinals as *ordered categories* in which the distances between the levels could be arbitrary.

### Booleans
A boolean (or *binary*) variable is a special case of a categorical variable in which the number of possible categories is strictly two. For example, to denote if a subject is tall or not you can use the `tall` and `not tall` categories respectively.

## Text
Text data are a product of language communication and can be viewed as a dense encoding of many sparse features. Initially, text blobs are imported as categorical features, however, they have little meaning as a category because the features are still encoded. Thus, import text blobs and use a [preprocessing](preprocessing.md) step to extract features such as word counts, weighted term frequencies, or word embeddings.

## Images
Images are represented as either the [GD](https://www.php.net/manual/en/book.image.php) resource type or a `GdImage` object. An image type is a special type that holds a reference to the data stored within the image file. For this reason, images must eventually be converted to a scalar type, such as the RGB color intensity values of each pixel, before they can be the input to a learning algorithm.

## Date/Time
There are a number of ways that date/time features can be represented in a dataset. One way is to discretize the value into days, months, and years using categories like `june`, `july`, `august`, and `2020`, `2021`. Date/times can also be represented as continuous features by converting them to a numerical timestamp.

## What about NULL?
Null values are often used to indicate the absence of a value, however since they do not give any information as to the type of variable that is missing, they cannot be used in a dataset. Instead, represent missing values as either the standard PHP math constant `NAN` for continuous features or use a special category (such as `?`) to denote missing categorical values.
