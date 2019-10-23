# Representing Your Data
Data are a first class citizen in Rubix ML. The library makes it easy to work with datasets through its [Dataset](./datasets/api.md) object, which is a specialized container for data that every learner can recognize and use. It allows you to validate, sort, randomize, split, fold, and describe your data among many other things. The most basic dataset includes a table (or *matrix*) of samples comprised of features which are scalar. Each sample is a sequential array with exactly the same number of elements. The column of a sample represents the value of a particular feature. The *dimensionality* of a sample is equal to the number of features it has. For example, the samples below are said to be 3-dimensional. You'll notice that samples an contain a heterogeneous mix of data types which we'll describe in detail in the next section. The important thing to note is that each feature column must only contain values of the same internal data type.

```php
$samples = [
    [0.1, 20, 'furry'],
    [2.0, -5, 'rough'],
    [0.001, -10, 'rough'],
];
```

### Internal Data Types
In addition to PHP's type system, Rubix ML adds a layer on top to distinguish types that are continuous (numerical), categorical (discrete), or some other type such as resource. Continuous features represent some *quantitative* property of the sample such as velocity, whereas, categorical features signal a *qualitative* property such as rough or furry. It is necessary to make the distinction between these data types in the context of machine learning because different learners are compatible with different data types. For example, the [Naive Bayes](./classifiers/naive-bayes.md) classifier is compatible with categorical features while [Gaussian Naive Bayes](./classifiers/gaussian-naive-bayes.md) is compatible with continuous ones - and [Random Forest](./classifiers/random-forest.md) is compatible with both. Throughout Rubix ML, categorical features are represented as strings whereas continuous data are represented as either integer or floating point numbers.

### PHP Representations
| Rubix ML Data Type | PHP Representation |
|---|---|
| Categorical | string |
| Continuous | int or float |
| Resource | resource |
| Other | object, array, bool, etc. |

### Boolean Values
Boolean type is simply a special case of a categorical variable where the number of categories is stricly two. Therefore, boolean features can easily be representated as a categorical variable in Rubix ML using strings. For example, to denote if someone is tall or not you can use either the `'tall'` or `'not tall'` categories respectively.

### Numeric Strings
Even though the standard PHP library function `is_numeric()` returns true for numeric strings such as "5" or "2.5", they are still considered categorical variables according to Rubix ML. This conveniently allows you to represent ordinal variables as ordered categories. For example, instead of the integers `1`, `2`, `3`, etc., which will be interpreted as a precise interval by a learner, you could use the strings `'1'`, `'2'`, `'3'`, etc. to signal ordinal values in which the precise *distance* between values could be arbitrary.

### Ints or Floats?
Since Rubix ML identifies both integer and floating point types as continuous data, you may be wondering which type to use to represent your features. The answer is you can use either, but the optimal usage comes down to what the data are representing and the precision of the measured features. For example, it might be better to represent the sale price of a house as an integer since a fraction of a unit of currency is negligible in that case. However, if you are representing a precise sensor measurement such as rotational acceleration from a gyroscope then a floating point number is better suited.

### What about NULL?
Null values are often used to indicate the absence of a value, however since it does not tell us the type of the variable that is missing, it is rather meaningless inside of Rubix ML as a datum. Instead, represent missing values as either `NaN` for continuous variables or a use separate category (such as `'?'`) just for missing values for categorical variables.