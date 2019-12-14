<?php

namespace Rubix\ML\Datasets\Extractors;

use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Datasets\Unlabeled;
use InvalidArgumentException;
use RuntimeException;
use Generator;

/**
 * NDJSON Array
 *
 * NDJSON or *Newline Delimited* JSON files contain rows of data encoded in Javascript Object
 * Notation (JSON) arrays. The format is similar to CSV but has the advantage of retaining data
 * type information at the cost of having a slightly heavier footprint.
 *
 * > **Note:** Empty rows will be ignored by the parser by default.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class NDJSONArray implements Extractor
{
    /**
     * The path to the NDJSON file.
     *
     * @var string
     */
    protected $path;

    /**
     * @param string $path
     * @throws \InvalidArgumentException
     */
    public function __construct(string $path)
    {
        if (is_file($path) and !is_readable($path)) {
            throw new InvalidArgumentException("File at $path is not readable.");
        }

        $this->path = $path;
    }

    /**
     * Extract and build an unlabeled dataset object from source.
     *
     * @param int $offset
     * @param int $limit
     * @return \Rubix\ML\Datasets\Unlabeled
     */
    public function extract(int $offset = 0, int $limit = PHP_INT_MAX) : Unlabeled
    {
        $records = $this->parseRecords($offset, $limit);

        $samples = iterator_to_array($records);

        return Unlabeled::build($samples);
    }

    /**
     * Extract and build a labeled dataset object from source.
     *
     * @param int $offset
     * @param int $limit
     * @return \Rubix\ML\Datasets\Labeled
     */
    public function extractWithLabels(int $offset = 0, int $limit = PHP_INT_MAX) : Labeled
    {
        $records = $this->parseRecords($offset, $limit);

        $samples = $labels = [];

        foreach ($records as $record) {
            $samples[] = array_slice($record, 0, -1);
            $labels[] = end($record);
        }

        return Labeled::build($samples, $labels);
    }

    /**
     * Read the records starting at the given offset and return them in an iterator.
     *
     * @param int $offset
     * @param int $limit
     * @throws \RuntimeException
     * @return \Generator
     */
    protected function parseRecords(int $offset, int $limit) : Generator
    {
        if (!$handle = fopen($this->path, 'r')) {
            throw new RuntimeException("Could not open file at {$this->path}.");
        }

        $line = $n = 0;

        while (true) {
            $row = fgets($handle);

            if (is_bool($row)) {
                break 1;
            }

            if (empty($row)) {
                continue 1;
            }

            if ($line >= $offset) {
                $record = json_decode($row);

                if (!is_array($record)) {
                    throw new RuntimeException('Non JSON array found'
                        . " at offset $line.");
                }

                yield $record;

                ++$n;

                if ($n >= $limit) {
                    break 1;
                }
            }

            ++$line;
        }
    }
}
