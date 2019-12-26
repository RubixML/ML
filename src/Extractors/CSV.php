<?php

namespace Rubix\ML\Extractors;

use InvalidArgumentException;
use RuntimeException;
use Generator;

/**
 * CSV
 *
 * A non-standard plain-text format that use newlines to delineate rows and a user-specified
 * delimiter (usually a comma) to separate the values of each column in a data table.
 * Comma-Separated Values (CSV) format is a common format but suffers from the disadvantage
 * of not being able to retain type information - thus, all data is imported as categorical
 * data (strings) by default.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class CSV implements Extractor
{
    /**
     * The path to the CSV file.
     *
     * @var string
     */
    protected $path;

    /**
     * The character that delineates the values of the columns of the data table.
     *
     * @var string
     */
    protected $delimiter;

    /**
     * Does the CSV document have a header as the first row?
     *
     * @var bool
     */
    protected $header;

    /**
     * The character used to enclose each column.
     *
     * @var string
     */
    protected $enclosure;

    /**
     * @param string $path
     * @param string $delimiter
     * @param bool $header
     * @param string $enclosure
     * @throws \InvalidArgumentException
     */
    public function __construct(
        string $path,
        string $delimiter = ',',
        bool $header = false,
        string $enclosure = ''
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

        if (!empty($enclosure) and strlen($enclosure) !== 1) {
            throw new InvalidArgumentException('Delimiter must be'
                . ' a single character, ' . strlen($enclosure) . ' given.');
        }

        $this->path = $path;
        $this->delimiter = $delimiter;
        $this->header = $header;
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
        $handle = fopen($this->path, 'r');
        
        if (!$handle) {
            throw new RuntimeException("Could not open $this->path.");
        }

        $line = 0;

        if ($this->header) {
            ++$line;

            $row = rtrim(fgets($handle) ?: '');

            $header = str_getcsv($row, $this->delimiter, $this->enclosure);
        }

        while (!feof($handle)) {
            ++$line;

            $row = rtrim(fgets($handle) ?: '');

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

        fclose($handle);
    }
}
