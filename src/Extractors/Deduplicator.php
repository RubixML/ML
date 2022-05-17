<?php

namespace Rubix\ML\Extractors;

use OkBloomer\BloomFilter;
use Traversable;

use function serialize;

/**
 * Deduplicator
 *
 * Removes duplicate records from a dataset while the records are in flight. Deduplicator uses a memory-efficient
 * Bloom filter to probabilistically identify records that have already been seen before.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Deduplicator implements Extractor
{
    /**
     * The base iterator.
     *
     * @var iterable<mixed[]>
     */
    protected iterable $iterator;

    /**
     * The Bloom filter.
     *
     * @var \OkBloomer\BloomFilter
     */
    protected BloomFilter $filter;

    /**
     * The number of records that have been dropped so far.
     *
     * @var int
     */
    protected int $dropped = 0;

    /**
     * @param iterable<mixed[]> $iterator
     * @param float $maxFalsePositiveRate
     * @param int|null $numHashes
     * @param int $layerSize
     */
    public function __construct(
        iterable $iterator,
        float $maxFalsePositiveRate = 0.001,
        ?int $numHashes = 4,
        int $layerSize = 32000000
    ) {
        $this->iterator = $iterator;
        $this->filter = new BloomFilter($maxFalsePositiveRate, $numHashes, $layerSize);
    }

    /**
     * Return the number of records that have been dropped so far.
     *
     * @return int
     */
    public function dropped() : int
    {
        return $this->dropped;
    }

    /**
     * Return an iterator for the records in the data table.
     *
     * @return \Generator<mixed[]>
     */
    public function getIterator() : Traversable
    {
        foreach ($this->iterator as $record) {
            $token = serialize($record);

            if ($this->filter->existsOrInsert($token)) {
                ++$this->dropped;

                continue;
            }

            yield $record;
        }
    }
}
