<?php

namespace Rubix\ML\Datasets\Extractors;

use League\Csv\Reader;
use League\Csv\Statement;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Datasets\Unlabeled;
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
     * @throws \RuntimeException
     */
    public function __construct(
        string $path,
        string $delimiter = ',',
        bool $header = true,
        ?string $enclosure = null
    ) {
        if (is_file($path) and !is_readable($path)) {
            throw new RuntimeException("File $path is not readable.");
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
     * Extract and build an unlabeled dataset object from source.
     *
     * @param int $offset
     * @param int|null $limit
     * @return \Rubix\ML\Datasets\Unlabeled
     */
    public function extract(int $offset = 0, ?int $limit = null) : Unlabeled
    {
        $statement = new Statement();
        
        if ($offset) {
            $statement = $statement->offset($offset);
        }

        if (isset($limit)) {
            $statement = $statement->limit($limit);
        }

        $records = $statement->process($this->reader);

        $samples = iterator_to_array($records);

        return Unlabeled::build($samples);
    }

    /**
     * Extract and build a labeled dataset object from source.
     *
     * @param int $offset
     * @param int|null $limit
     * @return \Rubix\ML\Datasets\Labeled
     */
    public function extractWithLabels(int $offset = 0, ?int $limit = null) : Labeled
    {
        $statement = new Statement();
        
        if ($offset) {
            $statement = $statement->offset($offset);
        }

        if (isset($limit)) {
            $statement = $statement->limit($limit);
        }

        $records = $statement->process($this->reader);

        $samples = $labels = [];

        foreach ($records as $record) {
            $samples[] = array_slice($record, 0, -1);
            $labels[] = end($record);
        }

        return Labeled::build($samples, $labels);
    }
}
