<?php

namespace Rubix\ML\Extractors;

use Rubix\ML\Exceptions\RuntimeException;
use Generator;
use PDO;

/**
 * SQL Table
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class SQLTable implements Extractor
{
    /**
     * The connection to the database.
     *
     * @var \PDO
     */
    protected $connection;

    /**
     * The columns to select from the SQL table.
     *
     * @var list<string>
     */
    protected $columns;

    /**
     * @param \PDO $connection
     * @param string $table
     * @param int $batchSize
     */
    public function __construct(PDO $connection, string $table, int $batchSize)
    {
        $this->connection = $connection;
        $this->table = $table;
        $this->batchSize = $batchSize;
    }

    /**
     * Return an iterator for the records in the data table.
     *
     * @return \Generator<mixed[]>
     */
    public function getIterator() : Generator
    {
        $offset = 0;

        while (true) {
            $query = "SELECT * FROM " . $this->table . " LIMIT $offset, " . $this->batchSize;

            $statement = $this->connection->prepare($query);

            $success = $statement->execute();

            if (!$success) {
                throw new RuntimeException('Could not execute SQL query.');
            }
            
            $rows = $statement->fetchAll();

            if (empty($rows)) {
                break;
            }

            yield from $rows;

            $offset += $this->batchSize;
        }
    }
}
