<?php

namespace Rubix\ML\NeuralNet;

use Rubix\ML\NeuralNet\Layers\Parametric;
use IteratorAggregate;
use ArrayIterator;

/**
 * Snapshot
 *
 * A snapshot represents the state of a nerual network at a moment in time.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 *
 * @implements IteratorAggregate<int, \Rubix\ML\NeuralNet\Layers\Layer>
 */
class Snapshot implements IteratorAggregate
{
    /**
     * The layer and parameter storage.
     *
     * @var array[]
     */
    protected $storage;

    /**
     * @param \Rubix\ML\NeuralNet\Network $network
     */
    public function __construct(Network $network)
    {
        $storage = [];

        foreach ($network->parametric() as $layer) {
            if ($layer instanceof Parametric) {
                $storage[] = [$layer, $layer->read()];
            }
        }

        $this->storage = $storage;
    }

    /**
     * Get an iterator for the snapshot.
     *
     * @return \ArrayIterator
     */
    public function getIterator()
    {
        return new ArrayIterator($this->storage);
    }
}
