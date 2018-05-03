<?php

namespace Rubix\Engine\NeuralNet\Layers;

use SplFixedArray;

abstract class Layer extends SplFixedArray
{
    /**
     * Activate the neurons in the layer and return an array of activations.
     *
     * @return array
     */
    public function fire() : array
    {
        $activations = [];

        for ($this->rewind(); $this->valid(); $this->next()) {
            $activations[] = $this->current()->output();
        }

        return $activations;
    }
}
