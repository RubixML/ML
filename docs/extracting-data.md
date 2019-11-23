# Extracting Data
Data will need to be loaded it into your ML project before it can become useful. There are many ways in which data can be stored, but the most common formats are either in text format such as CSV or in a database such as MySQL. More advanced online systems will have an ETL (*extract transform load*) pipeline set up to deliver the dataset in real-time or at regular intervals. The way in which your data is delivered makes no difference to Rubix ML as the library only cares about the data once it has already been loaded into memory in PHP. This gives the developer the freedom and flexibility to implement the data source to fit the scale of the problem and current infrastructure.

## CSV/TSV
One of the most common formats that you'll find smaller datasets in is comma-separated (CSV) or tab-separated (TSV) values files. Their popularity is largely due to their simplicity, interpretability, and ubiquity. A CSV file is a text file that contains a table with samples indicated by rows and the values of the features as columns separated either by a comma or tab. PHP has a number of built-in functions that allow you to access data stored in CSV format including `str_getcsv()` that parses a CSV string into an array.

**Example**

```php
$samples = str_getcsv(file_get_contents('dataset.csv'));
```

In addition, there are libraries such as the PHP League's [CSV Reader/Writer](https://csv.thephpleague.com/) that are highly efficient and make extracting data from CSV quick and easy.

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