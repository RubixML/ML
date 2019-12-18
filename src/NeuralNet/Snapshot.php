<?php

namespace Rubix\ML\NeuralNet;

use Rubix\ML\NeuralNet\Layers\Parametric;
use IteratorAggregate;
use Generator;

/**
 * Snapshot
 *
 * A snapshot represents the state of a nerual network at a moment in time.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 *
 * @implements IteratorAggregate<int, array>
 */
class Snapshot implements IteratorAggregate
{
    /**
     * The layer of the network.
     *
     * @var \Rubix\ML\NeuralNet\Layers\Layer[]
     */
    protected $layers;

    /**
     * The parameters cooresponding to each layer in the network at the
     * time of the snapshot.
     *
     * @var array[]
     */
    protected $params;

    /**
     * @param \Rubix\ML\NeuralNet\Network $network
     */
    public function __construct(Network $network)
    {
        $layers = $params = [];

        foreach ($network->parametric() as $layer) {
            if ($layer instanceof Parametric) {
                $layers[] = $layer;
                $params[] = $layer->read();
            }
        }

        $this->layers = $layers;
        $this->params = $params;
    }

    /**
     * Get an iterator over the layers and parameters of the snapshot.
     *
     * @return \Generator<array>
     */
    public function getIterator() : Generator
    {
        foreach ($this->layers as $i => $layer) {
            yield [$layer, $this->params[$i]];
        }
    }
}
