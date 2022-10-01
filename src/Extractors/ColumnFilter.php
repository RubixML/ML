<?php

namespace Rubix\ML\Extractors;

use Traversable;

/**
 * Column Filter
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class ColumnFilter implements Extractor
{
    /**
     * The base iterator.
     *
     * @var iterable<mixed[]>
     */
    protected iterable $iterator;

    /**
     * The string and/or integer keys of the columns to filter from the table.
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
        foreach ($this->iterator as $record) {
            foreach ($this->columns as $column) {
                if (isset($record[$column])) {
                    unset($record[$column]);
                }
            }

            yield $record;
        }
    }
}
