# Extracting Data
There are a number of ways to instantiate a new [Dataset](datasets/api.md) object, but all of them require the data to be loaded into memory first. Some common formats you'll find data in are structured plain-text such as CSV or NDJSON, or in a queryable database such as MySQL or MongoDB. No matter how your data are stored, you have the freedom and flexibility to implement the data source to fit your current infrastructure. To help make extraction simple for more common use cases, the library provides a number of [Extractor](extractors/api.md) objects. Extractors are iterators that let you loop over the records of a dataset in storage and can be used to instantiate a dataset object using the `fromIterator()` method.

## CSV
A common plain-text format for small to medium-sized datasets is [comma-separated values](https://en.wikipedia.org/wiki/Comma-separated_values) or CSV for short. A CSV file contains a table with individual samples indicated by rows and the values of the features stored in each column. Columns are separated by a *delimiter* such as the `,` or `;` character and may be enclosed on both ends with an optional *enclosure* such as `"`. The file can sometimes contain a header as the first row. CSV files have the advantage of being able to be processed line by line, however, their disadvantage is that type information cannot be inferred from the format. Thus, all CSV data are imported as categorical type (strings) by default.

**Example**

```csv
attitude,texture,sociability,rating,class
nice,furry,friendly,4,not monster
mean,furry,loner,-1.5,monster
```

The library provides the [CSV](extractors/csv.md) Extractor to help import data from the CSV format. We'll use it in conjunction with the `fromIterator()` method to instantiate a new dataset object. In the example below, In addition, we'll apply the [Numeric String Converter](transformers/numeric-string-converter.md) to the newly instantiated dataset object to convert the numeric data to the proper format immediately after instantiation.

```php
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Extractors\CSV;
use Rubix\ML\Transformers\NumericStringConverter;

$dataset = Labeled::fromIterator(new CSV('example.csv', true))
    ->apply(new NumericStringConverter());
```

We can check the number of records that were imported by calling the `numSamples()` method on the dataset object.

```php
echo $dataset->numSamples();
```

```
5000
```

## NDJSON
Another plain-text format called [NDJSON](http://ndjson.org/) or *Newline Delimited* Javascript Object Notation (JSON) can be considered a hybrid of both CSV and JSON. It contains rows of JSON arrays or objects delineated by a newline character (`\n` or `\r\n`). It has the advantage of retaining type information like JSON and can also be read into memory efficiently like CSV.

**Example**

```json
{"attitude":"nice","texture":"furry","sociability":"friendly","rating":4,"class":"not monster"}
{"attitude":"mean","texture":"furry","sociability":"loner","rating":-1.5,"class":"monster"}
{"attitude":"nice","texture":"rough","sociability":"friendly","rating":2.6,"class":"not monster"}
```

The [NDJSON](extractors/ndjson.md) extractor can be used to instantiate a new dataset object from a NDJSON file. Optionally, it can be combined with the standard PHP library's [Limit Iterator](https://www.php.net/manual/en/class.limititerator.php) to only load a portion of the data into memory. In the example below, we load the first 1,000 rows of data from an NDJSON file into an [Unlabeled](datasets/unlabeled.md) dataset.

```php
use Rubix\ML\Extractors\NDJSON;
use Rubix\ML\Datasets\Unlabeled;
use LimitIterator;

$extractor = new NDJSON('example.ndjson');

$iterator = new LimitIterator($extractor->getIterator(), 0, 1000);

$dataset = Unlabeled::fromIterator($iterator);
```

## SQL
Medium to large datasets will often be stored in an RDBMS (relational database management system) such as [MySQL](https://www.mysql.com), [PostgreSQL](https://www.postgresql.org)or [Sqlite](https://www.sqlite.org). Relational databases allow you to query large amounts of data on-the-fly and can be very flexible. PHP comes with robust relational database support through its [PDO](https://www.php.net/manual/en/book.pdo.php) interface. To iterate over the rows of an SQL table we provide an [SQL Table](extractors/sql-table.md) extractor uses the PDO interface under the hood. In the example below we'll wrap our SQL Table extractor in a [Column Picker](extractors/column-picker.md) to instantiate a new Unlabeled dataset object from a particular set of columns of the table.

```php
use Rubix\ML\Extractors\SQLTable;
use Rubix\ML\Extractors\ColumnPicker;
use Rubix\ML\Datasets\Unlabeled;
use PDO;

$connection = new PDO('sqlite:/example.sqlite');

$extractor = new ColumnPicker(new SQLTable($connection, 'patients'), [
    'age', 'gender', 'height', 'diagnosis',
]);

$dataset = Unlabeled::fromIterator($extractor);
```

If you need more control over your data pipeline then we recommend writing your own custom queries. The following example uses the PDO interface to execute a user-defined SQL query and instantiate a dataset object containing the same data as the example above. However, this method may be more efficient because it avoids querying for more data than you need.

```php
use Rubix\ML\Datasets\Unlabeled;
use PDO;

$pdo = new PDO('sqlite:/example.sqlite');

$query = $pdo->prepare('SELECT age, gender, height, diagnosis FROM patients');

$query->execute();

$samples = $query->fetchAll(PDO::FETCH_NUM);

$dataset = new Unlabeled($samples);
```

## Images
PHP offers a number of functions to import images as PHP resources such as `imagecreatefromjpeg()` and `imagecreatefrompng()` that come with the [GD](https://www.php.net/manual/en/book.image.php) extension. The example below imports the *.png* images in the `train` folder and labels them using part of their filename. The samples and labels are then put into a [Labeled](datasets/labeled.md) dataset using the `build()` factory method and then converted into raw color channel data by applying the [Image Vectorizer](transformers/image-vectorizer.md).

```php
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Transformers\ImageVectorizer;

$samples = $labels = [];

foreach (glob('train/*.png') as $file) {
    $samples[] = [imagecreatefrompng($file)];
    $labels[] = preg_replace('/[0-9]+_(.*).png/', '$1', basename($file));
}

$dataset = Labeled::build($samples, $labels)
    ->apply(new ImageVectorizer());
```

## Synthetic Datasets
Synthetic datasets are those that can be generated by one or more predefined formulas. In Rubix ML, we can generate synthetic datasets using [Generator](datasets/generators/api.md) objects. Generators are useful in educational settings and for supplementing a small dataset with more samples. To generate a labeled dataset using the [Half Moon](datasets/generators/half-moon.md) generator pass the number of records you wish to generate to the `generate()` method.

```php
use Rubix\ML\Datasets\Generators\HalfMoon;

$generator = new HalfMoon();

$dataset = $generator->generate(1000);
```

Now we can write the dataset to a CSV file and import it into our favorite plotting software.

```php
use Rubix\ML\Extractors\CSV;

$dataset->exportTo(new CSV('half-moon.csv'));
```

![Half Moon Dataset Scatterplot](https://github.com/RubixML/ML/blob/master/docs/images/half-moon-scatterplot.png?raw=true)
