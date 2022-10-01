<?php

namespace Rubix\ML\Extractors;

use Traversable;

/**
 * Concatenator
 *
 * Combines multiple iterators by concatenating the output of one iterator with the output of
 * the next iterator in the series.
 *
 * @category    Machine Learning
 * @package     Rubix\ML
 * @author      Andrew DalPino
 */
class Concatenator implements Extractor
{
    /**
     * An iterator of iterators.
     *
     * @var iterable<iterable<mixed[]>>
     */
    protected iterable $iterators;

    /**
     * @param iterable<iterable<mixed[]>> $iterators
     */
    public function __construct(iterable $iterators)
    {
        $this->iterators = $iterators;
    }

    /**
     * Return an iterator for the rows of a data table.
     *
     * @return \Generator<mixed[]>
     */
    public function getIterator() : Traversable
    {
        foreach ($this->iterators as $iterator) {
            foreach ($iterator as $record) {
                yield $record;
            }
        }
    }
}
