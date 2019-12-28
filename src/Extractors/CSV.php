<?php

namespace Rubix\ML\Extractors;

use InvalidArgumentException;
use IteratorAggregate;
use RuntimeException;
use Generator;

use function strlen;

/**
 * CSV
 *
 * A plain-text format that use newlines to delineate rows and a user-specified delimiter
 * (usually a comma) to separate the values of each column in a data table. Comma-Separated
 * Values (CSV) format is a common format but suffers from not being able to retain type
 * information - thus, all data is imported as categorical data (strings) by default.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 *
 * @implements IteratorAggregate<int, array>
 */
class CSV implements IteratorAggregate
{
    /**
     * The file handle.
     *
     * @var resource
     */
    protected $handle;

    /**
     * Does the CSV document have a header as the first row?
     *
     * @var bool
     */
    protected $header;

    /**
     * The character that delineates the values of the columns of the data table.
     *
     * @var string
     */
    protected $delimiter;

    /**
     * The character used to enclose each column.
     *
     * @var string
     */
    protected $enclosure;

    /**
     * @param string $path
     * @param bool $header
     * @param string $delimiter
     * @param string $enclosure
     * @throws \InvalidArgumentException
     * @throws \RuntimeException
     */
    public function __construct(
        string $path,
        bool $header = false,
        string $delimiter = ',',
        string $enclosure = ''
    ) {
        if (!is_file($path)) {
            throw new InvalidArgumentException("Path $path does not exist.");
        }
        
        if (!is_readable($path)) {
            throw new InvalidArgumentException("Path $path is not readable.");
        }

        $handle = fopen($path, 'r');
        
        if (!$handle) {
            throw new RuntimeException("Could not open $path.");
        }

        if (strlen($delimiter) !== 1) {
            throw new InvalidArgumentException('Delimiter must be'
                . ' a single character, ' . strlen($delimiter) . ' given.');
        }

        if (!empty($enclosure) and strlen($enclosure) !== 1) {
            throw new InvalidArgumentException('Delimiter must be'
                . ' a single character, ' . strlen($enclosure) . ' given.');
        }

        $this->handle = $handle;
        $this->header = $header;
        $this->delimiter = $delimiter;
        $this->enclosure = $enclosure;
    }

    /**
     * Return an iterator for the records in the data table.
     *
     * @throws \RuntimeException
     * @return \Generator<array>
     */
    public function getIterator() : Generator
    {
        rewind($this->handle);

        $line = 0;

        if ($this->header) {
            ++$line;

            $row = rtrim(fgets($this->handle) ?: '');

            $header = str_getcsv($row, $this->delimiter, $this->enclosure);
        }

        while (!feof($this->handle)) {
            ++$line;

            $row = rtrim(fgets($this->handle) ?: '');

            if (empty($row)) {
                continue 1;
            }

            $record = str_getcsv($row, $this->delimiter, $this->enclosure);

            if (isset($header)) {
                $record = array_combine($header, $record);
            }

            if (!$record) {
                throw new RuntimeException("Malformed CSV on line $line.");
            }

            yield $record;
        }
    }

    /**
     * Clean up the file pointer.
     */
    public function __destruct()
    {
        fclose($this->handle);
    }
}
