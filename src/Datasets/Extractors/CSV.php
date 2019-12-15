<?php

namespace Rubix\ML\Datasets\Extractors;

use League\Csv\Reader;
use League\Csv\Statement;
use Rubix\ML\Datasets\Extractors\Traits\Cursorable;
use InvalidArgumentException;

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
     * The CSV reader instance.
     *
     * @var \League\Csv\Reader
     */
    protected $reader;

    /**
     * @param string $path
     * @param string $delimiter
     * @param bool $header
     * @param string|null $enclosure
     * @throws \InvalidArgumentException
     */
    public function __construct(
        string $path,
        string $delimiter = ',',
        bool $header = true,
        ?string $enclosure = null
    ) {
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

        if (isset($enclosure) and strlen($enclosure) !== 1) {
            throw new InvalidArgumentException('Enclosure must be'
                . " a single character, $enclosure given.");
        }

        $reader = Reader::createFromPath($path)
            ->setDelimiter($delimiter)
            ->skipEmptyRecords();

        if ($header) {
            $reader->setHeaderOffset(0);
        }

        if ($enclosure) {
            $reader->setEnclosure($enclosure);
        }

        $this->reader = $reader;
    }

    /**
     * Read the records starting at the given offset and return them in an iterator.
     *
     * @return iterable
     */
    public function extract() : iterable
    {
        $statement = new Statement();
        
        return $statement->offset($this->offset)
            ->limit($this->limit)
            ->process($this->reader);
    }
}
