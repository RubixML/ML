<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Extractors/SQLTable.php">[source]</a></span>

# SQL Table
Efficiently iterates over the rows of a relational database table such as MySQL, PostgreSQL, and Sqlite.

!!! note
    This extractor imports the entire SQL table as-is. We recommend using your own SQL logic for more complex pipelines.

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
