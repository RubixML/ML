<?php

namespace Rubix\ML\Extractors;

use IteratorAggregate;

/**
 * @implements IteratorAggregate<int, array>
 */
abstract class Extractor implements IteratorAggregate
{
    /**
     * The row offset of the cursor.
     *
     * @var int
     */
    protected $offset = 0;

    /**
     * The maximum number of rows to return.
     *
     * @var int
     */
    protected $limit = PHP_INT_MAX;

    /**
     * Set the row offset of the cursor.
     *
     * @param int $offset
     * @return self
     */
    public function setOffset(int $offset) : self
    {
        $this->offset = $offset;

        return $this;
    }

    /**
     * Set the maximum number of rows to return.
     *
     * @param int $limit
     * @return self
     */
    public function setLimit(int $limit) : self
    {
        $this->limit = $limit;

        return $this;
    }

    /**
     * Read the records and return them in an iterator.
     *
     * @return iterable<array>
     */
    abstract public function extract() : iterable;

    /**
     * Get an iterator for the records in the data table.
     *
     * @return iterable<array>
     */
    public function getIterator() : iterable
    {
        return $this->extract();
    }
}
