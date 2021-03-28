<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Extractors/SQLTable.php">[source]</a></span>

# SQL Table
The SQL table extractor iterates over the rows of a relational database table. It works with the PHP Data Objects (PDO) interface to connect to a broad selection of databases such MySQL, PostgreSQL, and Sqlite.

!!! note
    This extractor requires the [PDO extension](https://www.php.net/manual/en/book.pdo.php).

!!! note
    The order in which the rows are iterated over is not guaranteed. Use a custom query with `ORDER BY` statement if ordering is important.

**Interfaces:** [Extractor](api.md)

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | connection | | PDO | The PDO connection to the database. |
| 2 | table | | string | The name of the table to select from. |
| 3 | batch size | 200 | int | The number of rows of the table to load in a single query. |

## Example
```php
use Rubix\ML\Extractors\SQLTable;
use PDO;

$connection = new PDO('sqlite:/example.sqlite');

$this->extractor = new SQLTable($connection, 'users', 200);
```

## Additional Methods
This extractor does not have any additional methods.
