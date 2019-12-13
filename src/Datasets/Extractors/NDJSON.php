<?php

namespace Rubix\ML\Datasets\Extractors;

use Rubix\ML\Datasets\Unlabeled;
use RuntimeException;

/**
 * NDJSON
 *
 * Newline Delimited JSON (NDJSON) files contain rows of Javascript Object Notation (JSON)
 * encoded data. The rows can either be JSON arrays with integer keys or objects with string
 * keys. One advantage NDJSON has over the CSV format is that it retains data type information.
 *
 * > **Note:** Empty rows will be ignored by the parser by default.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class NDJSON implements Extractor
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
     * @return \Rubix\ML\Datasets\Unlabeled
     */
    public function extract(int $offset = 0, ?int $limit = null) : Unlabeled
    {
        $ndjson = trim(file_get_contents($this->path) ?: '');

        $rows = explode(self::LINE_SEPARATOR, $ndjson);

        if ($offset or $limit) {
            $rows = array_slice($rows, $offset, $limit);
        }

        $samples = [];

        foreach ($rows as $row) {
            if (!empty($row)) {
                $samples[] = array_values(json_decode($row, true));
            }
        }

        return Unlabeled::build($samples);
    }
}
