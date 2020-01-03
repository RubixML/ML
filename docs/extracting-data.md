# Extracting Data
Data needs to be loaded into your project before it can become useful. Data can be stored in many formats, but the most common formats are either structured plain-text such as CSV or NDJSON or in a database such as MySQL or MongoDB. More advanced online systems will have an ETL (*extract transform load*) pipeline set up to deliver the dataset in real-time or at regular intervals. The way in which your data is delivered makes no difference to Rubix ML. You have the freedom and flexibility to implement the data source to fit the scale of the problem and your current infrastructure. In addition, the library provides [Extractor](extractors/api.md) objects to help automate more common use cases.

## CSV
A common format for small to medium-sized datasets is [comma-separated values](https://en.wikipedia.org/wiki/Comma-separated_values) (CSV). A CSV file is a plain-text file that contains a table with samples indicated by rows and the values of the features as columns. The columns are separated by a *delimiter* such as the `,` or `;` character and may be enclosed on both ends with an optional *enclosure* such as `"`. The file can sometimes contain a header as the first row which gives names to the feature columns. CSV files have the advantage of being able to be processed line by line. One disadvantage of CSV is that type information cannot be inferred from the format and thus all data is imported as categorical (strings) by default.

**Example**

```csv
attitude,texture,sociability,rating,class
nice,furry,friendly,4,not monster
mean,furry,loner,-1.5,monster
```

Rubix ML provides the [CSV](extractors/csv.md) extractor to help import data from the CSV format and can be used in conjunction with a Dataset's `fromIterator()` method to instantiate a new dataset object. In the example below, we'll apply the [Numeric String Converter](transformers/numeric-string-converter.md) to the newly instantiated dataset object to convert the numeric data to the proper format.

**Example**

```php
use Rubix\ML\Extractors\CSV;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Transformers\NumericStringConverter;

$dataset = Labeled::fromIterator(new CSV('example.csv', true))
    ->apply(new NumericStringConverter());
```

## JSON
Javascript Object Notation (JSON) is a standardized lightweight plain-text format that is used to represent structured data such as objects and arrays. The rows of a dataset can either be represented as a sequential array or an object with keyed properties. Since it is possible to derive the original data type from the JSON format, JSON files have the advantage of importing the data as the proper type. One downside of the JSON format, however, is that the entire document must be read all at once.

**Example**

```json
[
    {
        "attitude": "nice",
        "texture": "furry",
        "sociability": "friendly",
        "rating": 4,
        "class": "not monster"
    },
    [
        "mean",
        "furry",
        "loner",
        -1.5,
        "monster"
    ]
]
```

The [JSON](extractors/json.md) extractor in Rubix ML handles loading data from JSON files.

**Example**

```php
use Rubix\ML\Extractors\JSON;
use Rubix\ML\Datasets\Labeled;

$dataset = Labeled::fromIterator(new JSON('example.json'));
```

## NDJSON
Another popular plain-text format is a hybrid of CSV and JSON called [NDJSON](http://ndjson.org/) or *Newline Delimited* Javascript Object Notation (JSON). It contains rows of JSON arrays or objects delineated by a newline character (`\n` or `\r\n`). It has the advantage of retaining type information like JSON and can also be read into memory efficiently like CSV.

**Example**

```ndjson
{"attitude":"nice","texture":"furry","sociability":"friendly","rating":4,"class":"not monster"}
["mean","furry","loner",-1.5,"monster"]
```

The [NDJSON](extractors/ndjson.md) extractor can be used to instantiate a new dataset object from a NDJSON file. Optionally, it can be combined with the standard PHP library's [Limit Iterator](https://www.php.net/manual/en/class.limititerator.php) to only load a portion of the data into memory. In the example below, we load the first 1,000 rows of data from an NDJSON file into an [Unlabeled](datasets/unlabeled.md) dataset.

**Example**

```php
use Rubix\ML\Extractors\NDJSON;
use Rubix\ML\Datasets\Unlabeled;
use LimitIterator;

$extractor = new NDJSON('example.ndjson');

$iterator = new LimitIterator($extractor->getIterator(), 0, 1000);

$dataset = Unlabeled::fromIterator($iterator);
```

## SQL
Medium to large datasets will often be stored in an RDBMS (relational database management system) like [MySQL](https://www.mysql.com) or [SQLite](https://www.sqlite.org). Relational databases allow you to query large amounts of data on-the-fly and can be very flexible. PHP comes with robust relational database support through its [PDO](https://www.php.net/manual/en/book.pdo.php) interface. The following example uses PDO and the `fetchAll()` method to return the first 1,000 rows fo data from the `patients` table. Then we'll load those sample into an [Unlabeled](datasets/unlabeled.md) dataset object.

**Example**

```php
use Rubix\ML\Datasets\Unlabeled;

$pdo = new PDO('mysql:dbname=example;host=127.0.0.1');

$query = $pdo->prepare('SELECT age, gender, height, diagnosis FROM patients LIMIT 1000');

$query->execute();

$samples = $query->fetchAll();

$dataset = new Unlabeled($samples);
```

## Images
Some machine learning tasks such as image recognition involve data that are stored in image files. PHP offers a number of functions to import images as PHP resources such as `imagecreatefromjpeg()` and `imagecreatefrompng()` that come with the [GD](https://www.php.net/manual/en/book.image.php) extension. The example below loops over all the `.png` files in the `train` folder, imports the images as resources and labels them with the part of their filename after the underscore. The samples are then converted into raw color channel data by apply the [Image Vectorizer](transformers/image-vectorizer.md) to the newly instantiated dataset object.

**Example**

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