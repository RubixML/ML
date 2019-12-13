<?php

namespace Rubix\ML\Datasets\Extractors;

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
    protected const NEWLINE_REGEX = '/((\r?\n)|(\r\n?))/';

    /**
     * The path to the CSV file.
     *
     * @var string
     */
    protected $path;

    /**
     * The character that delineates a new column.
     *
     * @var string
     */
    protected $delimiter;

    /**
     * The character used to enclose the value of a column.
     *
     * @var string
     */
    protected $enclosure;

    /**
     * @param string $path
     * @param string $delimiter
     * @param string $enclosure
     * @throws \InvalidArgumentException
     * @throws \RuntimeException
     */
    public function __construct(
        string $path,
        string $delimiter = ',',
        string $enclosure = ''
    ) {
        if (is_file($path) and !is_readable($path)) {
            throw new RuntimeException("File $path is not readable.");
        }

        if (strlen($delimiter) !== 1) {
            throw new InvalidArgumentException('Delimiter must be'
                . " a single character, $delimiter given.");
        }

        if (strlen($enclosure) > 1) {
            throw new InvalidArgumentException('Enclosure must be'
                . ' less than or equal to 1 character.');
        }

        $this->path = $path;
        $this->delimiter = $delimiter;
        $this->enclosure = $enclosure;
    }

    /**
     * Extract and build a dataset object from source.
     *
     * @param int $offset
     * @param int|null $limit
     * @return \Rubix\ML\Datasets\Unlabeled
     */
    public function extract(int $offset = 0, ?int $limit = null) : Unlabeled
    {
        $csv = trim(file_get_contents($this->path) ?: '');

        $rows = preg_split(self::NEWLINE_REGEX, $csv) ?: [];

        if ($offset or $limit) {
            $rows = array_slice($rows, $offset, $limit);
        }

        $samples = [];

        foreach ($rows as $row) {
            if (!empty($row)) {
                $samples[] = str_getcsv($row, $this->delimiter, $this->enclosure);
            }
        }

        return Unlabeled::build($samples);
    }
}
