<?php

namespace Rubix\ML\Extractors;

use InvalidArgumentException;
use RuntimeException;
use Generator;

use function is_null;

/**
 * NDJSON
 *
 * NDJSON or *Newline Delimited* JSON files contain rows of data encoded in Javascript Object
 * Notation (JSON) arrays or objects. The format is like a mix of JSON and CSV and has the
 * advantage of retaining data type information and being cursorable (read line by line).
 *
 * > **Note:** Empty rows will be ignored by the parser by default.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class NDJSON implements Extractor
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
        if (!is_file($path)) {
            throw new InvalidArgumentException("Path $path does not exist.");
        }
        
        if (!is_readable($path)) {
            throw new InvalidArgumentException("Path $path is not readable.");
        }

        $this->path = $path;
    }

    /**
     * Return an iterator for the records in the data table.
     *
     * @throws \RuntimeException
     * @return \Generator<array>
     */
    public function getIterator() : Generator
    {
        $handle = fopen($this->path, 'r');
        
        if (!$handle) {
            throw new RuntimeException("Could not open $this->path.");
        }

        $line = 0;

        while (!feof($handle)) {
            ++$line;

            $row = rtrim(fgets($handle) ?: '');

            if (empty($row)) {
                continue 1;
            }
            
            $record = json_decode($row, true);

            if (is_null($record)) {
                throw new RuntimeException("Malformed JSON on line $line.");
            }

            yield $record;
        }

        fclose($handle);
    }
}
