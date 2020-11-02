<?php

namespace Rubix\ML\NeuralNet;

use Rubix\ML\NeuralNet\Layers\Parametric;
use Rubix\ML\Exceptions\InvalidArgumentException;

/**
 * Snapshot
 *
 * A snapshot represents the state of a neural network at a moment in time.
 *
 * @internal
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Snapshot
{
    /**
     * The parametric layers of the network.
     *
     * @var \Rubix\ML\NeuralNet\Layers\Parametric[]
     */
    protected $layers;

    /**
     * The parameters corresponding to each layer in the network at the time of the snapshot.
     *
     * @var array[]
     */
    protected $parameters;

    /**
     * @param \Rubix\ML\NeuralNet\Network $network
     */
    public static function take(Network $network) : self
    {
        $layers = $parameters = [];

        foreach ($network->layers() as $layer) {
            if ($layer instanceof Parametric) {
                $params = [];

                foreach ($layer->parameters() as $key => $parameter) {
                    $params[$key] = clone $parameter;
                }

                $layers[] = $layer;
                $parameters[] = $params;
            }
        }

        return new self($layers, $parameters);
    }

    /**
     * @param \Rubix\ML\NeuralNet\Layers\Parametric[] $layers
     * @param array[] $parameters
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     */
    public function __construct(array $layers, array $parameters)
    {
        if (count($layers) !== count($parameters)) {
            throw new InvalidArgumentException('Number of layers'
                . ' and parameter groups must be equal');
        }

        $this->layers = $layers;
        $this->parameters = $parameters;
    }

    /**
     * Restore the network parameters.
     */
    public function restore() : void
    {
        foreach ($this->layers as $i => $layer) {
            $layer->restore($this->parameters[$i]);
        }
    }
}
