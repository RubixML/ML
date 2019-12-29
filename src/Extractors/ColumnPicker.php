<?php

namespace Rubix\ML\Extractors;

use Traversable;
use Generator;

/**
 * Column Picker
 *
 * An extractor that wraps another iterator and selects and rearranges the columns of the data
 * table according to the user-specified keys. The key of a column may either be a string or a
 * column number (integer) depending on the base iterator.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class ColumnPicker implements Extractor
{
    /**
     * The base iterator.
     *
     * @var \Traversable<array>
     */
    protected $iterator;

    /**
     * The keys of the columns to iterate over.
     *
     * @var (string|int)[]
     */
    protected $keys;

    /**
     * @param \Traversable<array> $iterator
     * @param (string|int)[] $keys
     */
    public function __construct(Traversable $iterator, array $keys)
    {
        $this->iterator = $iterator;
        $this->keys = $keys;
    }

    /**
     * Return an iterator for the records in the data table.
     *
     * @return \Generator<array>
     */
    public function getIterator() : Generator
    {
        foreach ($this->iterator as $record) {
            $temp = [];

            foreach ($this->keys as $key) {
                if (isset($record[$key])) {
                    $temp[$key] = $record[$key];
                }
            }

            yield $temp;
        }
    }
}
