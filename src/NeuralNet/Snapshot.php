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
     * @var Parametric[]
     */
    protected array $layers;

    /**
     * The parameters corresponding to each layer in the network at the time of the snapshot.
     *
     * @var list<Parameter[]>
     */
    protected array $parameters;

    /**
     * @param Network $network
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
     * @param Parametric[] $layers
     * @param list<Parameter[]> $parameters
     * @throws InvalidArgumentException
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
