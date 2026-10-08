<?php

namespace Rubix\ML\Extractors;

use Generator;
use Traversable;

use Rubix\ML\Exceptions\InvalidArgumentException;

use function is_iterable;
use function get_debug_type;

/**
 * Repeater
 *
 * Repeats the records of a data table a given number of times while they are in flight. Repeater
 * re-traverses the base iterator for every pass rather than buffering the records in memory, which
 * keeps memory usage constant even for very large or infinite streams.
 *
 * > **Note:** The base iterator must be re-iterable. That means it has to be an array, an
 * IteratorAggregate such as another Extractor, a rewindable Iterator, or a callable factory that
 * returns a fresh iterator for each pass. Raw Generators cannot be traversed twice by PHP and are
 * therefore rejected when more than one pass is requested - pass the function that returns a
 * fresh generator instead.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Repeater implements Extractor
{
    /**
     * The base iterator or a callable factory that returns one.
     *
     * @var iterable<mixed[]>|callable(): mixed
     */
    protected mixed $iterator;

    /**
     * The total number of passes over the records.
     *
     * @var int
     */
    protected int $repetitions;

    /**
     * @param iterable<mixed[]>|callable(): mixed $iterator
     * @param int $repetitions
     * @throws InvalidArgumentException
     */
    public function __construct(iterable|callable $iterator, int $repetitions)
    {
        if ($repetitions < 1) {
            throw new InvalidArgumentException('Number of repetitions must be'
                . " greater than 0, $repetitions given.");
        }

        if ($repetitions > 1 and $iterator instanceof Generator) {
            throw new InvalidArgumentException('Base iterator must be re-iterable when'
                . ' repeating, a Generator can only be traversed once. Pass the callable'
                . ' that returns a fresh iterator for each pass instead.');
        }

        $this->iterator = $iterator;
        $this->repetitions = $repetitions;
    }

    /**
     * Return an iterator for the records in the data table.
     *
     * @throws InvalidArgumentException
     * @return Generator<mixed[]>
     */
    public function getIterator() : Traversable
    {
        for ($i = 0; $i < $this->repetitions; ++$i) {
            $iterator = $this->iterator;

            if (!is_iterable($iterator)) {
                $iterator = $iterator();
            }

            if (!is_iterable($iterator)) {
                throw new InvalidArgumentException('Iterator factory must return an'
                    . ' iterable type, ' . get_debug_type($iterator) . ' given.');
            }

            foreach ($iterator as $record) {
                yield $record;
            }
        }
    }
}
