<?php

namespace Rubix\ML\Datasets\Extractors;

use Rubix\ML\Datasets\Extractors\Traits\Cursorable;
use InvalidArgumentException;
use RuntimeException;

use function is_null;

/**
 * NDJSON
 *
 * NDJSON or *Newline Delimited* JSON files contain rows of data encoded in Javascript Object
 * Notation (JSON) arrays or objects. The format is similar to CSV but has the advantage of
 * being standardized and retaining data type information at the cost of having a slightly
 * heavier footprint.
 *
 * > **Note:** Empty rows will be ignored by the parser by default.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class NDJSON implements Extractor
{
    use Cursorable;

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
            throw new InvalidArgumentException("File at $path does not exist.");
        }
        
        if (!is_readable($path)) {
            throw new InvalidArgumentException("File at $path is not readable.");
        }

        $this->path = $path;
    }

    /**
     * Read the records starting at the given offset and return them in an iterator.
     *
     * @throws \RuntimeException
     * @return iterable<array>
     */
    public function extract() : iterable
    {
        $handle = fopen($this->path, 'r');
        
        if (!$handle) {
            throw new RuntimeException("Could not open file at {$this->path}.");
        }

        $line = $n = 0;

        while (!feof($handle)) {
            $row = rtrim(fgets($handle) ?: '');

            if (empty($row)) {
                continue 1;
            }

            ++$line;

            if ($line > $this->offset) {
                $record = json_decode($row);

                if (is_null($record)) {
                    throw new RuntimeException("Malformed JSON on line $line.");
                }

                if (is_object($record)) {
                    $record = array_values((array) $record);
                }

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
