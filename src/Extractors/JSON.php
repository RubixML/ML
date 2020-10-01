<?php

namespace Rubix\ML\Extractors;

use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;
use Generator;
use Rubix\ML\Other\Helpers\JSON as JSONHelper;

/**
 * JSON
 *
 * Javascript Object Notation is a standardized lightweight plain-text representation that
 * is widely used. JSON has the advantage of retaining type information, however since the
 * entire JSON blob is read on load, it is not cursorable like CSV or NDJSON.
 *
 * References:
 * [1] T. Bray. (2014). The JavaScript Object Notation (JSON) Data Interchange Format.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class JSON implements Extractor
{
    /**
     * The path to the JSON file.
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
     * @throws \Rubix\ML\Exceptions\RuntimeException
     * @return \Generator<mixed[]>
     */
    public function getIterator() : Generator
    {
        $data = file_get_contents($this->path);

        if (!$data) {
            throw new RuntimeException("Could not open $this->path.");
        }

        yield from JSONHelper::decode($data);
    }
}
