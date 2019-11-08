# Extracting Data
Data will need to be loaded it into your ML project before it can become useful. There are many ways in which data can be stored, but the most common formats are either in text format such as CSV or in a database such as MySQL. More advanced online systems will have an ETL (*extract transform load*) pipeline set up to deliver the dataset in real-time or at regular intervals. The way in which your data is delivered makes no difference to Rubix ML as it only cares about the data once it has already been loaded into memory in PHP. This gives the developer the freedom and flexibility to implement the data source to fit the scale of the problem and current infrastructure.

### CSV/TSV
One of the most common formats that you'll find smaller datasets in is comma-separated (CSV) or tab-separated (TSV) version files. Their popularity is largely due to their simplicity, interpretability, and ubiquity. A CSV file is a text file that contains a table with samples indicated by rows and the values of the features as columns. PHP has a number of built-in functions that allow you to access data stored in CSV format including `str_getcsv()` that parses a CSV string into an array.

**Example**

```php
$samples = str_getcsv(file_get_contents('dataset.csv'));
```

In addition, there are libraries such as the PHP League's [CSV Reader/Writer](https://csv.thephpleague.com/) that are highly efficient and make extracting data from CSV quick and easy.

### Database
For bigger datasets, the data will often be stored in some type of database such as an RDBMS (relational database management system) like MySQL or an object store such as MongoDB. PHP already comes built-in with great database support such as [PDO](https://www.php.net/manual/en/book.pdo.php) for relational databases and other extensions for other popular databases. In addition, the PHP community has developed a healthy ecosystem of DBALs (Database Abstraction Layers) such as [Doctrine DBAL](https://www.doctrine-project.org/projects/dbal.html) and ORMs (Object Relational Mappers) such as [Eloquent](https://laravel.com/docs/5.8/eloquent) that make it even easier to get the data you need.