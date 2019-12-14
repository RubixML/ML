# Extracting Data
Data will need to be loaded it into your project before it can become useful. There are many ways in which data can be stored, but the most common formats are either in plain-text format such as CSV or NDJSON and in a database such as MySQL or MongoDB. More advanced online systems will have an ETL (*extract transform load*) pipeline set up to deliver the dataset in real-time or at regular intervals. The way in which your data is delivered makes no difference to Rubix ML. Thus, you have the freedom and flexibility to implement the data source to fit the scale of the problem and current infrastructure. In addition, the library provides  [Extractor](datasets/extractors/api.md) objects to help automate more common use cases.

## CSV
One of the most common formats that you'll find smaller datasets in is comma-separated (CSV) or tab-separated (TSV) values files. Their popularity is largely due to their simplicity, interpretability, and ubiquity. A CSV file is a text file that contains a table with samples indicated by rows and the values of the features as columns separated either by a comma or tab. Rubix ML provides the [CSV](datasets/extractors/csv.md) extractor to help import the data. You can always import your data manually or with the help of other tools such as the PHP League's [CSV Reader/Writer](https://csv.thephpleague.com/). The disadvantage of CSV is that data type information cannot be inferred from the format and thus all data is imported as categorical (strings) by default. The library provides the [Numeric String Converter](transformers/numeric-string-converter.md) to handle transforming the data into the proper format after the dataset has been extracted.

**Example**

```php
use Rubix\ML\Datasets\Extractors\CSV;
use Rubix\ML\Transformers\NumericStringConverter;

$extractor = new CSV('example.csv', ',');

$dataset = $extractor->extract()->apply(new NumericStringConverter());
```

## NDJSON
Another popular plain-text format is a hybrid of CSV and JSON called NDJSON or *Newline Delimited* Javascript Object Notation (JSON). It contains rows of JSON arrays or objects delineated by a newline character. Since it is possible to derive the original data type from JSON, NDJSON files have the advantage of importing the data with the proper type foregoing the need for conversion. The [NDJSON](datasets/extractors/ndjson.md) extractor is designed to help you import data in the NDJSON format.

**Example**

```php
use Rubix\ML\Datasets\Extractors\NDJSON;

$extractor = new NDJSON('example.ndjson');

$dataset = $extractor->extract(0, 5000);
```

## Database
For bigger datasets, the data will often be stored in some type of database such as an RDBMS (relational database management system) like MySQL or an object store such as MongoDB. PHP already comes built-in with great database support such as [PDO](https://www.php.net/manual/en/book.pdo.php) for relational databases and other extensions for other popular databases. In addition, the PHP community has developed a healthy ecosystem of DBALs (Database Abstraction Layers) such as [Doctrine DBAL](https://www.doctrine-project.org/projects/dbal.html) and ORMs (Object Relational Mappers) such as [Eloquent](https://laravel.com/docs/5.8/eloquent) that make it even easier to get the data you need. The following example uses PDO and the `fetchAll()` method to return a 2-d array of samples.

**Example**

```php
$pdo = new PDO('mysql:dbname=example;host=127.0.0.1');

$query = $pdo->prepare('SELECT age, gender, height FROM patients');

$query->execute();

$samples = $query->fetchAll();
```

## Images
The PHP language offers a number of functions to import images as PHP resources including `imagecreatefromjpeg()` and `imagecreatefrompng()` that come with the [GD](https://www.php.net/manual/en/book.image.php) extension. The example below loops over all the `.png` files in the `train` folder, imports the images as PHP resource types and labels them with the part of their filename after the underscore.

**Example**

```php
$samples = $labels = [];

foreach (glob('train/*.png') as $file) {
    $samples[] = [imagecreatefrompng($file)];
    $labels[] = preg_replace('/[0-9]+_(.*).png/', '$1', basename($file));
}
```

> **Note:** Images as they come are not compatible with any learner, but they can be converted into compatible raw color channel data once they have been imported into your project using the [Image Vectorizer](transformers/image-vectorizer.md).