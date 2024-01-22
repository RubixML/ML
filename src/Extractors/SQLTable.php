<?php

namespace Rubix\ML\Extractors;

use Rubix\ML\Specifications\ExtensionIsLoaded;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;
use Traversable;
use PDO;

use function Rubix\ML\iterator_first;
use function count;
use function array_keys;

/**
 * SQL Table
 *
 * The SQL table extractor iterates over the rows of a relational database table. It works with
 * the PHP Data Objects (PDO) interface to connect to a broad selection of databases such MySQL,
 * PostgreSQL, and Sqlite.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class SQLTable implements Extractor
{
    /**
     * The PDO connection to the database.
     *
     * @var PDO
     */
    protected PDO $connection;

    /**
     * The name of the table to select from.
     *
     * @var string
     */
    protected string $table;

    /**
     * The number of rows of the table to load in a single query.
     *
     * @var int
     */
    protected int $batchSize;

    /**
     * @param PDO $connection
     * @param string $table
     * @param int $batchSize
     * @throws InvalidArgumentException
     */
    public function __construct(PDO $connection, string $table, int $batchSize = 256)
    {
        ExtensionIsLoaded::with('PDO')->check();

        if (empty($table)) {
            throw new InvalidArgumentException('Table name cannot be empty.');
        }

        if ($batchSize < 1) {
            throw new InvalidArgumentException('Batch size must be'
                . " greater than 0, $batchSize given.");
        }

        $this->connection = $connection;
        $this->table = $connection->quote($table);
        $this->batchSize = $batchSize;
    }

    /**
     * Return the column titles of the data table.
     *
     * @return array<string|int>
     */
    public function header() : array
    {
        return array_keys(iterator_first($this));
    }

    /**
     * Return an iterator for the records in the data table.
     *
     * @return \Generator<mixed[]>
     */
    public function getIterator() : Traversable
    {
        $query = "SELECT * FROM {$this->table} LIMIT :offset, {$this->batchSize}";

        $statement = $this->connection->prepare($query);

        $offset = 0;

        $statement->bindParam(':offset', $offset);

        do {
            $success = $statement->execute();

            if (!$success) {
                throw new RuntimeException('There was a problem executing the SQL statement.');
            }

            $rows = $statement->fetchAll(PDO::FETCH_ASSOC);

            if (empty($rows)) {
                break;
            }

            yield from $rows;

            $more = count($rows) >= $this->batchSize;

            $offset += $this->batchSize;
        } while ($more);
    }
}
