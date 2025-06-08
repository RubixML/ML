<?php

namespace Rubix\ML\Extractors;

use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;
use Traversable;
use Generator;

use function Rubix\ML\iterator_first;
use function is_dir;
use function is_file;
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
final class CSV implements Extractor, Exporter
{
    /**
     * The path to the file on disk.
     */
    private string $path;

    /**
     * Does the CSV document have a header as the first row?
     */
    private bool $header;

    /**
     * The character that delineates the values of the columns of the data table.
     */
    private string $delimiter;

    /**
     * The character used to enclose a cell that contains a delimiter in the body.
     */
    private string $enclosure;

    /**
     * The character used as an escape character (one character only). Defaults as a backslash.
     */
    private string $escape;

    /**
     * @param non-empty-string $path
     * @param non-empty-string $delimiter
     * @param non-empty-string $enclosure
     * @param non-empty-string $escape
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

        $this->validateCharacter($delimiter, 'Delimiter');
        $this->validateCharacter($enclosure, 'Enclosure');
        $this->validateCharacter($escape, 'Escape character');

        $this->path = $path;
        $this->header = $header;
        $this->delimiter = $delimiter;
        $this->enclosure = $enclosure;
        $this->escape = $escape;
    }

    /**
     * Validate that a parameter is a single character.
     * 
     * @param non-empty-string $char
     * @param non-empty-string $name
     * @throws InvalidArgumentException
     */
    private function validateCharacter(string $char, string $name): void
    {
        if (strlen($char) !== 1) {
            throw new InvalidArgumentException(
                sprintf('%s must be a single character, %d given.', $name, strlen($char))
            );
        }
    }

    /**
     * Return the column titles of the data table.
     *
     * @return array<string|int>
     */
    public function header(): array
    {
        return array_keys(iterator_first($this->getIterator()));
    }

    /**
     * Export an iterable data table.
     *
     * @param iterable<array<mixed>> $iterator
     * @throws RuntimeException
     */
    public function export(iterable $iterator): void
    {
        $this->ensureWritable();

        $handle = fopen($this->path, 'wb');
        if (!$handle) {
            throw new RuntimeException('Could not open file pointer.');
        }

        try {
            $line = 1;

            if ($this->header) {
                $header = array_keys(iterator_first($iterator));
                $this->writeCsvLine($handle, $header, $line);
                ++$line;
            }

            foreach ($iterator as $row) {
                $this->writeCsvLine($handle, $row, $line);
                ++$line;
            }
        } finally {
            fclose($handle);
        }
    }

    /**
     * Ensure the file is writable.
     * 
     * @throws RuntimeException
     */
    private function ensureWritable(): void
    {
        if (is_file($this->path) && !is_writable($this->path)) {
            throw new RuntimeException("Path {$this->path} is not writable.");
        }

        if (!is_file($this->path) && !is_writable(dirname($this->path))) {
            throw new RuntimeException("Path {$this->path} is not writable.");
        }
    }

    /**
     * Write a line to CSV file.
     * 
     * @param resource $handle
     * @param array<mixed> $data
     * @param positive-int $line
     * @throws RuntimeException
     */
    private function writeCsvLine($handle, array $data, int $line): void
    {
        $length = fputcsv($handle, $data, $this->delimiter, $this->enclosure, $this->escape);
        if ($length === false) {
            throw new RuntimeException("Could not write data on line $line.");
        }
    }

    /**
     * Return an iterator for the records in the data table.
     *
     * @throws RuntimeException
     * @return Generator<array<mixed>>
     */
    public function getIterator(): Traversable
    {
        $this->ensureReadable();

        $handle = fopen($this->path, 'rb');
        if (!$handle) {
            throw new RuntimeException('Could not open file pointer.');
        }

        try {
            $line = 1;
            $header = $this->header ? $this->readHeader($handle, $line) : null;

            while (!feof($handle)) {
                $record = fgetcsv($handle, 0, $this->delimiter, $this->enclosure, $this->escape);
                
                if (empty($record)) {
                    continue;
                }

                if ($header !== null) {
                    $record = array_combine($header, $record);
                    if (!is_array($record)) {
                        throw new RuntimeException("Malformed record on line $line.");
                    }
                }

                yield $record;
                ++$line;
            }
        } finally {
            fclose($handle);
        }
    }

    /**
     * Ensure the file is readable.
     * 
     * @throws RuntimeException
     */
    private function ensureReadable(): void
    {
        if (!is_file($this->path)) {
            throw new RuntimeException("Path {$this->path} is not a file.");
        }

        if (!is_readable($this->path)) {
            throw new RuntimeException("Path {$this->path} is not readable.");
        }
    }

    /**
     * Read the header row from CSV.
     * 
     * @param resource $handle
     * @param positive-int $line
     * @return array<string>
     * @throws RuntimeException
     */
    private function readHeader($handle, int &$line): array
    {
        $header = fgetcsv($handle, 0, $this->delimiter, $this->enclosure, $this->escape);
        if (!$header) {
            throw new RuntimeException("Header not found on line $line.");
        }
        ++$line;
        
        return $header;
    }
}
