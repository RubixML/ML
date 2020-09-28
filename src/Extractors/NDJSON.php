<?php

namespace Rubix\ML\Extractors;

use InvalidArgumentException;
use RuntimeException;
use Generator;
use Rubix\ML\Other\Helpers\JSON;

/**
 * NDJSON
 *
 * NDJSON or *Newline Delimited* JSON files contain rows of data encoded in Javascript Object
 * Notation (JSON) arrays or objects. The format is like a mix of JSON and CSV and has the
 * advantage of retaining data type information and being read into memory incrementally.
 *
 * > **Note:** Empty lines are ignored by the parser.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class NDJSON implements Extractor
{
    /**
     * The file handle.
     *
     * @var resource
     */
    protected $handle;

    /**
     * @param string $path
     * @throws \InvalidArgumentException
     * @throws \RuntimeException
     */
    public function __construct(string $path)
    {
        if (!is_file($path)) {
            throw new InvalidArgumentException("Path $path does not exist.");
        }

        if (!is_readable($path)) {
            throw new InvalidArgumentException("Path $path is not readable.");
        }

        $handle = fopen($path, 'r');

        if (!$handle) {
            throw new RuntimeException("Could not open $path.");
        }

        $this->handle = $handle;
    }

    /**
     * Clean up the file pointer.
     */
    public function __destruct()
    {
        fclose($this->handle);
    }

    /**
     * Return an iterator for the records in the data table.
     *
     * @throws \RuntimeException
     * @return \Generator<mixed[]>
     */
    public function getIterator() : Generator
    {
        rewind($this->handle);

        $line = 0;

        while (!feof($this->handle)) {
            $data = rtrim(fgets($this->handle) ?: '');

            ++$line;

            if (empty($data)) {
                continue 1;
            }

            try {
                yield JSON::decode($data);
            } catch (RuntimeException $e) {
                throw new RuntimeException(
                    "JSON Error on line: $line. (" . $e->getMessage() . ')',
                    $e->getCode(),
                    $e
                );
            }
        }
    }
}
