<?php

namespace Rubix\ML\Datasets\Extractors;

use Rubix\ML\Datasets\Labeled;
use RuntimeException;

/**
 * NDJSON With Labels
 *
 * Newline Delimited JSON with Labels uses the last column of an NDJSON data table as the
 * value of the dataset's labels.
 *
 * > **Note:** Empty rows will be ignored by the parser by default.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class NDJSONWithLabels implements Extractor
{
    protected const LINE_SEPARATOR = "\n";

    /**
     * The path to the NDJSON file.
     *
     * @var string
     */
    protected $path;

    /**
     * @param string $path
     * @throws \RuntimeException
     */
    public function __construct(string $path)
    {
        if (is_file($path) and !is_readable($path)) {
            throw new RuntimeException("File $path is not readable.");
        }

        $this->path = $path;
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
        $ndjson = trim(file_get_contents($this->path) ?: '');

        $rows = explode(self::LINE_SEPARATOR, $ndjson);

        if ($offset or $limit) {
            $rows = array_slice($rows, $offset, $limit);
        }

        $samples = $labels = [];

        foreach ($rows as $row) {
            if (!empty($row)) {
                $data = json_decode($row, true);

                $samples[] = array_slice($data, 0, -1);
                $labels[] = end($data);
            }
        }

        return Labeled::build($samples, $labels);
    }
}
