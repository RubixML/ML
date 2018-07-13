<?php

namespace Rubix\ML\NeuralNet;

use Rubix\ML\NeuralNet\Layers\Parametric;
use SplObjectStorage;

class Snapshot extends SplObjectStorage
{
    /**
     * @param  array  $layers
     * @return void
     */
    public function __construct(array $layers = [])
    {
        foreach ($layers as $layer) {
            if ($layer instanceof Parametric) {
                $this->attach($layer, $layer->read());
            }
        }
    }
}
