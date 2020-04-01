<?php

namespace Rubix\ML\NeuralNet;

use Rubix\ML\NeuralNet\Layers\Parametric;

/**
 * Snapshot
 *
 * A snapshot represents the state of a nerual network at a moment in time.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Snapshot
{
    /**
     * The layer of the network.
     *
     * @var \Rubix\ML\NeuralNet\Layers\Parametric[]
     */
    protected $layers;

    /**
     * The parameters corresponding to each layer in the network at the
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

        foreach ($network->layers() as $layer) {
            if ($layer instanceof Parametric) {
                $temp = [];

                foreach ($layer->parameters() as $key => $param) {
                    $temp[$key] = clone $param;
                }

                $layers[] = $layer;
                $params[] = $temp;
            }
        }
        
        $this->layers = $layers;
        $this->params = $params;
    }

    /**
     * Restore the network parameters.
     */
    public function restore() : void
    {
        foreach ($this->layers as $i => $layer) {
            $layer->restore($this->params[$i]);
        }
    }
}
