<?php

namespace Rubix\ML\Extractors;

use Rubix\ML\Exceptions\RuntimeException;
use Traversable;

/**
 * Column Picker
 *
 * An extractor that wraps another iterator and selects and reorders the columns of the data
 * table according to the keys specified by the user. The key of a column may either be a
 * string or a column number (integer) depending on the way the columns are indexed in the
 * base iterator.
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
     * @var iterable<mixed[]>
     */
    protected iterable $iterator;

    /**
     * The string and/or integer keys of the columns to pick and reorder from the table.
     *
     * @var list<string|int>
     */
    protected array $columns;

    /**
     * @param iterable<mixed[]> $iterator
     * @param (string|int)[] $columns
     */
    public function __construct(iterable $iterator, array $columns)
    {
        $this->iterator = $iterator;
        $this->columns = array_values($columns);
    }

    /**
     * Return an iterator for the records in the data table.
     *
     * @return \Generator<mixed[]>
     */
    public function getIterator() : Traversable
    {
        foreach ($this->iterator as $i => $record) {
            $picked = [];

            foreach ($this->columns as $column) {
                if (!isset($record[$column])) {
                    throw new RuntimeException("Column '$column' not found"
                        . " at row offset $i.");
                }

                $picked[$column] = $record[$column];
            }

            yield $picked;
        }
    }
}
