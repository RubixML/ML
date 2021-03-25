<?php

namespace Rubix\ML\Extractors;

use Rubix\ML\Exceptions\InvalidArgumentException;
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
     * The PDO connection to the database.
     *
     * @var \PDO
     */
    protected $connection;

    /**
     * The name of the table to select from.
     *
     * @var string
     */
    protected $table;

    /**
     * The number of rows of the table to load in a single query.
     *
     * @var int
     */
    protected $batchSize;

    /**
     * @param \PDO $connection
     * @param string $table
     * @param int $batchSize
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     */
    public function __construct(PDO $connection, string $table, int $batchSize = 200)
    {
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
     * Return an iterator for the records in the data table.
     *
     * @return \Generator<mixed[]>
     */
    public function getIterator() : Generator
    {
        $query = "SELECT * FROM {$this->table} LIMIT :offset, {$this->batchSize}";

        $statement = $this->connection->prepare($query);

        $offset = 0;

        $statement->bindParam(':offset', $offset);

        while (true) {
            $success = $statement->execute();

            if (!$success) {
                throw new RuntimeException('There was a problem executing the SQL statement.');
            }

            $rows = $statement->fetchAll(PDO::FETCH_ASSOC);

            if (empty($rows)) {
                break;
            }

            yield from $rows;

            $offset += $this->batchSize;
        }
    }
}
