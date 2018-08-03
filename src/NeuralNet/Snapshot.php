<?php

namespace Rubix\ML\NeuralNet;

use Rubix\ML\NeuralNet\Network;
use Rubix\ML\NeuralNet\Layers\Parametric;
use SplObjectStorage;

class Snapshot extends SplObjectStorage
{
    /**
     * Take a snapshot
     *
     * @param  \Rubix\ML\NeuralNet\Network  $network
     * @return self
     */
    public static function take(Network $network) : self
    {
        $snapshot = new self();

        foreach ($network->parametric() as $layer) {
            if ($layer instanceof Parametric) {
                $snapshot->attach($layer, $layer->read());
            }
        }

        return $snapshot;
    }
}
