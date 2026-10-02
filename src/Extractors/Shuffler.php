<?php

namespace Rubix\ML\Extractors;

use Traversable;

use function shuffle;
use function rand;
use function count;

/**
 * Shuffler
 *
 * An extractor that wraps another iterator and randomizes the order of the records of the
 * data table while they are in flight. Shuffler uses a bounded shuffle buffer under the
 * hood to produce a random ordering of the records without holding more than a specified
 * number of records in memory at the same time.
 *
 * > **Note:** When the number of records in the stream is greater than the buffer size,
 * the resulting ordering is not guaranteed to be uniformly random since doing so would
 * require holding the entire stream in memory at once.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Shuffler implements Extractor
{
    /**
     * The base iterator.
     *
     * @var iterable<mixed[]>
     */
    protected iterable $iterator;

    /**
     * The maximum number of records to hold in memory at a time.
     *
     * @var int
     */
    protected int $bufferSize;

    /**
     * @param iterable<mixed[]> $iterator
     * @param int $bufferSize
     */
    public function __construct(iterable $iterator, int $bufferSize = 256)
    {
        if ($bufferSize < 1) {
            throw new \Rubix\ML\Exceptions\InvalidArgumentException('Buffer size must be'
                . " greater than 0, $bufferSize given.");
        }

        $this->iterator = $iterator;
        $this->bufferSize = $bufferSize;
    }

    /**
     * Return an iterator for the records in the data table.
     *
     * @return \Generator<mixed[]>
     */
    public function getIterator() : Traversable
    {
        $buffer = [];

        foreach ($this->iterator as $record) {
            $n = count($buffer);

            if ($n < $this->bufferSize) {
                $buffer[] = $record;

                continue;
            }

            $i = rand(0, $n - 1);

            yield $buffer[$i];

            $buffer[$i] = $record;
        }

        shuffle($buffer);

        yield from $buffer;
    }
}
