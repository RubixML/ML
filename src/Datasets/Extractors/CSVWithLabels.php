<?php

namespace Rubix\ML\Datasets\Extractors;

use Rubix\ML\Datasets\Labeled;
use InvalidArgumentException;
use RuntimeException;

/**
 * CSV With Labels
 *
 * A version of the CSV extractor where the last column of the data table is taken as
 * the values for the label of a labeled dataset.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class CSVWithLabels implements Extractor
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
     * @return \Rubix\ML\Datasets\Labeled
     */
    public function extract(int $offset = 0, ?int $limit = null) : Labeled
    {
        $csv = trim(file_get_contents($this->path) ?: '');

        $rows = preg_split(self::NEWLINE_REGEX, $csv) ?: [];

        if ($offset or $limit) {
            $rows = array_slice($rows, $offset, $limit);
        }

        $samples = $labels = [];

        foreach ($rows as $row) {
            if (!empty($row)) {
                $data = str_getcsv($row, $this->delimiter, $this->enclosure);

                $samples[] = array_slice($data, 0, -1);
                $labels[] = end($data);
            }
        }

        return Labeled::build($samples, $labels);
    }
}
