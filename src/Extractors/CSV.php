<?php

namespace Rubix\ML\Extractors;

use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;
use Traversable;

use function Rubix\ML\iterator_first;
use function is_dir;
use function is_file;
use function is_array;
use function is_readable;
use function is_writable;
use function fopen;
use function fgetcsv;
use function fputcsv;
use function fclose;
use function array_combine;
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
 * [1] Y. Shafranovich. (2005). Common Format and MIME Type for Comma-Separated Values (CSV) Files.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class CSV implements Extractor, Exporter
{
    /**
     * The path to the file on disk.
     *
     * @var non-empty-string
     */
    protected string $path;

    /**
     * Does the CSV document have a header as the first row?
     *
     * @var bool
     */
    protected bool $header;

    /**
     * The character that delineates the values of the columns of the data table.
     *
     * @var non-empty-string
     */
    protected string $delimiter;

    /**
     * The character used to enclose a cell that contains a delimiter in the body.
     *
     * @var non-empty-string
     */
    protected string $enclosure;

    /**
     * The character used as an escape character (one character only). Defaults as a backslash.
     *
     * @var non-empty-string
     */
    protected string $escape;

    /**
     * @param string $path
     * @param bool $header
     * @param string $delimiter
     * @param string $enclosure
     * @param string $escape
     * @throws InvalidArgumentException
     */
    public function __construct(
        string $path,
        bool $header = false,
        string $delimiter = ',',
        string $enclosure = '"',
        string $escape = '\\'
    ) {
        if (empty($path)) {
            throw new InvalidArgumentException('Path cannot be empty.');
        }

        if (is_dir($path)) {
            throw new InvalidArgumentException('Path must be to a file, folder given.');
        }

        if (strlen($delimiter) !== 1) {
            throw new InvalidArgumentException('Delimiter must be'
                . ' a single character, ' . strlen($delimiter) . ' given.');
        }

        if (strlen($enclosure) !== 1) {
            throw new InvalidArgumentException('Enclosure must be'
                . ' a single character, ' . strlen($enclosure) . ' given.');
        }

        if (strlen($escape) !== 1) {
            throw new InvalidArgumentException('Escape character must be'
                . ' a single character, ' . strlen($escape) . ' given.');
        }

        $this->path = $path;
        $this->header = $header;
        $this->delimiter = $delimiter;
        $this->enclosure = $enclosure;
        $this->escape = $escape;
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
     * Export an iterable data table.
     *
     * @param iterable<mixed[]> $iterator
     * @throws RuntimeException
     */
    public function export(iterable $iterator) : void
    {
        if (is_file($this->path) and !is_writable($this->path)) {
            throw new RuntimeException("Path {$this->path} is not writable.");
        }

        if (!is_file($this->path) and !is_writable(dirname($this->path))) {
            throw new RuntimeException("Path {$this->path} is not writable.");
        }

        $handle = fopen($this->path, 'w');

        if (!$handle) {
            throw new RuntimeException('Could not open file pointer.');
        }

        $line = 1;

        if ($this->header) {
            $header = array_keys(iterator_first($iterator));

            $length = fputcsv($handle, $header, $this->delimiter, $this->enclosure, $this->escape);

            if ($length === false) {
                throw new RuntimeException("Could not write header on line $line.");
            }

            ++$line;
        }

        foreach ($iterator as $row) {
            $length = fputcsv($handle, $row, $this->delimiter, $this->enclosure, $this->escape);

            if ($length === false) {
                throw new RuntimeException("Could not write row on line $line.");
            }

            ++$line;
        }

        fclose($handle);
    }

    /**
     * Return an iterator for the records in the data table.
     *
     * @throws RuntimeException
     * @return \Generator<mixed[]>
     */
    public function getIterator() : Traversable
    {
        if (!is_file($this->path)) {
            throw new RuntimeException("Path {$this->path} is not a file.");
        }

        if (!is_readable($this->path)) {
            throw new RuntimeException("Path {$this->path} is not readable.");
        }

        $handle = fopen($this->path, 'r');

        if (!$handle) {
            throw new RuntimeException('Could not open file pointer.');
        }

        $line = 1;

        if ($this->header) {
            $header = fgetcsv($handle, 0, $this->delimiter, $this->enclosure, $this->escape);

            if (!$header) {
                throw new RuntimeException("Header not found on line $line.");
            }

            ++$line;
        }

        while (!feof($handle)) {
            $record = fgetcsv($handle, 0, $this->delimiter, $this->enclosure, $this->escape);

            if (empty($record)) {
                continue;
            }

            if (isset($header)) {
                $record = array_combine($header, $record);

                if (!is_array($record)) {
                    throw new RuntimeException("Malformed record on line $line.");
                }
            }

            yield $record;

            ++$line;
        }

        fclose($handle);
    }
}
