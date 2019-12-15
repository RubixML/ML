<?php

namespace Rubix\ML\Datasets\Extractors;

use Rubix\ML\Datasets\Extractors\Traits\Cursorable;
use InvalidArgumentException;
use RuntimeException;

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
    use Cursorable;

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
     * The character used to enclose each column.
     *
     * @var string
     */
    protected $enclosure;

    /**
     * @param string $path
     * @param string $delimiter
     * @param string $enclosure
     * @throws \InvalidArgumentException
     */
    public function __construct(string $path, string $delimiter = ',', string $enclosure = '')
    {
        if (!is_file($path)) {
            throw new InvalidArgumentException("File at $path does not exist.");
        }
        
        if (!is_readable($path)) {
            throw new InvalidArgumentException("File at $path is not readable.");
        }

        if (strlen($delimiter) !== 1) {
            throw new InvalidArgumentException('Delimiter must be'
                . " a single character, $delimiter given.");
        }

        if (strlen($enclosure) > 1) {
            throw new InvalidArgumentException('Enclosure must be'
                . " a single character, $enclosure given.");
        }

        $this->path = $path;
        $this->delimiter = $delimiter;
        $this->enclosure = $enclosure;
    }

    /**
     * Read the records starting at the given offset and return them in an iterator.
     *
     * @throws \RuntimeException
     * @return iterable
     */
    public function extract() : iterable
    {
        $handle = fopen($this->path, 'r');
        
        if (!$handle) {
            throw new RuntimeException("Could not open file at {$this->path}.");
        }

        $line = $n = 0;

        while (!feof($handle)) {
            $row = fgets($handle);

            if (empty($row)) {
                continue 1;
            }

            ++$line;

            if ($line > $this->offset) {
                $record = str_getcsv($row, $this->delimiter, $this->enclosure);

                yield $record;

                ++$n;

                if ($n >= $this->limit) {
                    break 1;
                }
            }
        }

        fclose($handle);
    }
}
