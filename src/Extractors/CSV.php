<?php

namespace Rubix\ML\Extractors;

use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;
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
 * > **Note:** This implementation of CSV is based on the definition in RFC 4180.
 *
 * References:
 * [1] Y. Shafranovich. (2005). Common Format and MIME Type for Comma-Separated Values (CSV)
 * Files.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class CSV implements Extractor
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
     * The character used to enclose a cell that contains a delimiter in the body.
     *
     * @var string
     */
    protected $enclosure;

    /**
     * @param string $path
     * @param bool $header
     * @param string $delimiter
     * @param string $enclosure
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     * @throws \Rubix\ML\Exceptions\RuntimeException
     */
    public function __construct(
        string $path,
        bool $header = false,
        string $delimiter = ',',
        string $enclosure = '"'
    ) {
        if (!is_file($path)) {
            throw new InvalidArgumentException("Path $path does not exist.");
        }

        if (!is_readable($path)) {
            throw new InvalidArgumentException("Path $path is not readable.");
        }

        if (strlen($delimiter) !== 1) {
            throw new InvalidArgumentException('Delimiter must be'
                . ' a single character, ' . strlen($delimiter) . ' given.');
        }

        if (strlen($enclosure) !== 1) {
            throw new InvalidArgumentException('Enclosure must be'
                . ' a single character, ' . strlen($enclosure) . ' given.');
        }

        $handle = fopen($path, 'r');

        if (!$handle) {
            throw new RuntimeException("Could not open $path.");
        }

        $this->handle = $handle;
        $this->header = $header;
        $this->delimiter = $delimiter;
        $this->enclosure = $enclosure;
    }

    /**
     * Clean up the file pointer.
     */
    public function __destruct()
    {
        fclose($this->handle);
    }

    /**
     * Return an iterator for the records in the data table.
     *
     * @throws \Rubix\ML\Exceptions\RuntimeException
     * @return \Generator<mixed[]>
     */
    public function getIterator() : Generator
    {
        rewind($this->handle);

        $line = 0;

        if ($this->header) {
            $header = fgetcsv($this->handle, 0, $this->delimiter, $this->enclosure);

            if (!$header) {
                throw new RuntimeException('Header not found on the first line.');
            }

            ++$line;
        }

        while (!feof($this->handle)) {
            $record = fgetcsv($this->handle, 0, $this->delimiter, $this->enclosure);

            ++$line;

            if (empty($record)) {
                continue;
            }

            if (isset($header)) {
                $record = array_combine($header, $record);
            }

            if (!$record) {
                throw new RuntimeException("Malformed record on line $line.");
            }

            yield $record;
        }
    }
}
