<?php

namespace Rubix\ML\Extractors;

use Rubix\ML\Helpers\JSON;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;
use Generator;

use function is_dir;
use function is_file;
use function is_readable;
use function is_writable;
use function fopen;
use function fgets;
use function fputs;
use function fclose;
use function rtrim;

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
class NDJSON implements Extractor, Writable
{
    /**
     * The path to the file on disk.
     *
     * @var string
     */
    protected $path;

    /**
     * @param string $path
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     */
    public function __construct(string $path)
    {
        if (empty($path)) {
            throw new InvalidArgumentException('Path cannot be empty.');
        }

        if (is_dir($path)) {
            throw new InvalidArgumentException('Path must be to a file, folder given.');
        }

        $this->path = $path;
    }

    /**
     * Write an iterable data table to disk.
     *
     * @param iterable<mixed[]> $iterator
     * @throws \Rubix\ML\Exceptions\RuntimeException
     */
    public function write(iterable $iterator) : void
    {
        if (is_file($this->path) and !is_writable($this->path)) {
            throw new RuntimeException("Path {$this->path} is not writable.");
        }

        if (!is_file($this->path) and !is_writable(dirname($this->path))) {
            throw new RuntimeException("Path {$this->path} is not writable.");
        }

        $handle = fopen($this->path, 'w');

        if (!$handle) {
            throw new RuntimeException('Could not open file pointer.');
        }

        $line = 0;

        foreach ($iterator as $row) {
            $length = fputs($handle, JSON::encode($row) . PHP_EOL);

            ++$line;

            if (!$length) {
                throw new RuntimeException("Could not write row on line $line.");
            }
        }

        fclose($handle);
    }

    /**
     * Return an iterator for the records in the data table.
     *
     * @throws \Rubix\ML\Exceptions\RuntimeException
     * @return \Generator<mixed[]>
     */
    public function getIterator() : Generator
    {
        if (!is_file($this->path)) {
            throw new InvalidArgumentException("Path {$this->path} is not a file.");
        }

        if (!is_readable($this->path)) {
            throw new InvalidArgumentException("Path {$this->path} is not readable.");
        }

        $handle = fopen($this->path, 'r');

        if (!$handle) {
            throw new RuntimeException('Could not open file pointer.');
        }

        $line = 0;

        while (!feof($handle)) {
            $data = rtrim(fgets($handle) ?: '');

            ++$line;

            if (empty($data)) {
                continue;
            }

            try {
                yield JSON::decode($data);
            } catch (RuntimeException $exception) {
                throw new RuntimeException(
                    "JSON Error on line $line: {$exception->getMessage()}",
                    $exception->getCode(),
                    $exception
                );
            }
        }

        fclose($handle);
    }
}
