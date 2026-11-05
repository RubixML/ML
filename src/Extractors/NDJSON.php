<?php

namespace Rubix\ML\Extractors;

use Rubix\ML\Helpers\JSON;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;
use Traversable;

use function Rubix\ML\enumerate;
use function is_dir;
use function is_file;
use function is_readable;
use function is_writable;
use function fopen;
use function fgets;
use function fputs;
use function fclose;
use function fread;
use function fseek;
use function filesize;
use function clearstatcache;
use function rtrim;

use const JSON_PRESERVE_ZERO_FRACTION;

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
class NDJSON implements Extractor, Exporter
{
    /**
     * The path to the file on disk.
     *
     * @var string
     */
    protected string $path;

    /**
     * @param string $path
     * @throws InvalidArgumentException
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
     * Export an iterable data table.
     *
     * @param iterable<mixed[]> $iterator
     * @param bool $overwrite
     * @throws RuntimeException
     */
    public function export(iterable $iterator, bool $overwrite = false) : void
    {
        clearstatcache(true, $this->path);

        if (is_file($this->path) and !is_writable($this->path)) {
            throw new RuntimeException("Path {$this->path} is not writable.");
        }

        if (!is_file($this->path) and !is_writable(dirname($this->path))) {
            throw new RuntimeException("Path {$this->path} is not writable.");
        }

        $needsSeparator = false;

        if (!$overwrite and is_file($this->path) and filesize($this->path) > 0) {
            $existing = fopen($this->path, 'r');

            if ($existing) {
                fseek($existing, -1, SEEK_END);

                $last = fread($existing, 1);

                fclose($existing);

                if ($last !== "\n" and $last !== "\r") {
                    $needsSeparator = true;
                }
            }
        }

        $handle = fopen($this->path, $overwrite ? 'w' : 'a');

        if (!$handle) {
            throw new RuntimeException('Could not open file pointer.');
        }

        $firstRow = true;

        foreach (enumerate($iterator, 1) as $line => $row) {
            if ($firstRow and $needsSeparator) {
                if (fputs($handle, PHP_EOL) === false) {
                    throw new RuntimeException('Row separator could not be written.');
                }
            }

            $firstRow = false;

            $data = JSON::encode($row, JSON_PRESERVE_ZERO_FRACTION) . PHP_EOL;

            $length = fputs($handle, $data);

            if ($length === false) {
                throw new RuntimeException("Could not write row on line $line.");
            }
        }

        fclose($handle);
    }

    /**
     * Return an iterator for the records in the data table.
     *
     * @throws RuntimeException
     * @return \Generator<mixed[]>
     */
    public function getIterator() : Traversable
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

        $line = 1;

        while (!feof($handle)) {
            $data = rtrim(fgets($handle) ?: '');

            if (empty($data)) {
                ++$line;

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

            ++$line;
        }

        fclose($handle);
    }
}
