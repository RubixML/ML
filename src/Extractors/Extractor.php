<?php

namespace Rubix\ML\Extractors;

use IteratorAggregate;

/**
 * @implements IteratorAggregate<int, array>
 */
abstract class Extractor implements IteratorAggregate
{
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
