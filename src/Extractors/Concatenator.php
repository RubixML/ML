<?php

namespace Rubix\ML\Extractors;

use Traversable;

use function array_values;

/**
 * Concatenator
 *
 * Concatenates the output of multiple extractors.
 *
 * @category    Machine Learning
 * @package     Rubix\ML
 * @author      Andrew DalPino
 */
class Concatenator implements Extractor
{
    /**
     * A list of iterators.
     *
     * @var list<iterable<mixed[]>>
     */
    protected array $iterators;

    /**
     * @param array<iterable<mixed[]>> $iterators
     */
    public function __construct(array $iterators)
    {
        $this->iterators = array_values($iterators);
    }

    /**
     * Return an iterator for the sequences in a dataset.
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
